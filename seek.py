import os
from datetime import timedelta

from openai import OpenAI
from tqdm import tqdm

import gpt_lit_reviewer.scholar_api as sa
from gpt_lit_reviewer.gpt import Arbiter
from gpt_lit_reviewer.seeker import (
    filterByImpact, verboseUnion, amendAbstracts,
    showAndSaveResults, ratePaper,
)

from load_key import loadKey

STAGES = ['understanding', 'grounding']

def main(api_key: str):
    print('main()...')

    seed_papers = [
        ('MU-LLaMA', 'arXiv:2308.11276', 40, 1), 
        ('LLark',    'arXiv:2310.07160', 40, 1), 
        ('Kosmos-2', 'arXiv:2306.14824', 80, 0), 
    ]

    def prompt(stage: str):
        return f'''
Evaluate whether the paper fits the criterion.

<criterion>
{dict(
    understanding = 'Music understanding via finetuning an LLM or a foundation model. Music understanding include question answering, classification, tag detection, captioning, etc.',
    grounding = 'Bringing grounding to text LLMs. Just cross-modal info fusion is not enough. It has to mechanistically ground the text LLM to world content. E.g. in Kosmos-2, the text LLM refers to the images via bounding boxes, making it one step ahead of baselines that feed the image as a whole to the text LLM.',
)[stage]}
</criterion>

<paper>
%s
</paper>

Does the above paper fit the criterion? Answer "Yes" or "No", using exactly one single word.
'''.strip()

    scholarApi = sa.ScholarAPI(timedelta(days=1))

    all_impactful = []
    for desc, paper_id, cites_per_year_threshold, grace_period in seed_papers:
        for neighborType in sa.NeighborType:
            papers = scholarApi.getPaperNeighbors(neighborType, paper_id, fields=[
                sa.PAPERID, sa.TITLE, sa.ABSTRACT, sa.CITATIONCOUNT, sa.YEAR, 
                sa.EXTERNALIDS, 
            ], limit=1000)
            print('# of', neighborType.value, desc, ':', len(papers))
            impactful = [*filterByImpact(
                papers, 
                cites_per_year_threshold=cites_per_year_threshold, 
                grace_period=grace_period,
            )]
            print(f'impactful % = {len(impactful) / len(papers):.0%}')
            all_impactful.append(impactful)
    
    union = verboseUnion(all_impactful)
    
    if input('Proceed? y/n >').lower() != 'y':
        print('aborted')
        return

    amendAbstracts(union)

    GPT_MODEL = "gpt-4o-mini"

    print('Creating OpenAI client...')
    client = OpenAI(api_key=api_key)
    print('ok')

    allRatedPapers = {k: [] for k in STAGES}
    arbiter = Arbiter(client, timedelta(weeks=6))
    os.makedirs('./results', exist_ok=True)
    for stage, ratedPapers in allRatedPapers.items():
        for paper in tqdm(union, desc=stage):
            print()
            print(paper[sa.TITLE])
            score = ratePaper(
                arbiter, GPT_MODEL, prompt(stage), paper,
            )
            ratedPapers.append((paper, score))
            relevance = format(score, '.3%')
            print(f'{relevance = }')

    for stage, ratedPapers in allRatedPapers.items():
        showAndSaveResults(ratedPapers, f'./results/{stage}.txt', f'./results/{stage}.csv')

if __name__ == '__main__':
    main(api_key = loadKey())
