STAGES = ['understanding', 'grounding']

ours = [
    'FiLMu: A Multimodal Music-to-Text Model for Fine-Grained Understanding and Grounding',
    'We present FiLMu, a multimodal music-to-text framework for fine-grained and segment-level music analysis. While prior models focus on global descriptions, FiLMu introduces timestamp-aligned annotations and a structured HTML-like format that links specific audio segments with natural language. Built on LLaMA3-8B and fine-tuned using LoRA, FiLMu uses a trainable projector to bridge audio and language representations. Our structured data format unifies multiple MIR tasks and exposes fine-grained correspondences between music and text, revealing the potential of large language models for music understanding. FiLMu supports detailed segment-level analysis, comparison across multiple audio sections, and flexible querying of musical attributes within a single framework.', 
]

papers = {}

papers['understanding'] = r'''
Music Understanding LLaMA: Advancing Text-to-Music Generation with Question Answering and Captioning
mullama
2023
Text-to-music generation (T2M-Gen) faces a major obstacle due to the scarcity
of large-scale publicly available music datasets with natural language
captions. To address this, we propose the Music Understanding LLaMA (MU-LLaMA),
capable of answering music-related questions and generating captions for music
files. Our model utilizes audio representations from a pretrained MERT model to
extract music features. However, obtaining a suitable dataset for training the
MU-LLaMA model remains challenging, as existing publicly accessible audio
question answering datasets lack the necessary depth for open-ended music
question answering. To fill this gap, we present a methodology for generating
question-answer pairs from existing audio captioning datasets and introduce the
MusicQA Dataset designed for answering open-ended music-related questions. The
experiments demonstrate that the proposed MU-LLaMA model, trained on our
designed MusicQA dataset, achieves outstanding performance in both music
question answering and music caption generation across various metrics,
outperforming current state-of-the-art (SOTA) models in both fields and
offering a promising advancement in the T2M-Gen research field.

LLark: A Multimodal Instruction-Following Language Model for Music
llark
2023
Music has a unique and complex structure which is challenging for both expert humans and existing AI systems to understand, and presents unique challenges relative to other forms of audio. We present LLark, an instruction-tuned multimodal model for \emph{music} understanding. We detail our process for dataset creation, which involves augmenting the annotations of diverse open-source music datasets and converting them to a unified instruction-tuning format. We propose a multimodal architecture for LLark, integrating a pretrained generative model for music with a pretrained language model. In evaluations on three types of tasks (music understanding, captioning, reasoning), we show that LLark matches or outperforms existing baselines in music understanding, and that humans show a high degree of agreement with its responses in captioning and reasoning tasks. LLark is trained entirely from open-source music data and models, and we make our training code available along with the release of this paper. Additional results and audio examples are at this https URL, and our source code is available at this https URL . 
<comment>LLark cited MU-LLaMA.</comment>

DeepResonance: Enhancing Multimodal Music Understanding via Music-centric Multi-way Instruction Tuning
mao2025deepresonance
2025
Recent advancements in music large language models (LLMs) have significantly
improved music understanding tasks, which involve the model's ability to
analyze and interpret various musical elements. These improvements primarily
focused on integrating both music and text inputs. However, the potential of
incorporating additional modalities such as images, videos and textual music
features to enhance music understanding remains unexplored. To bridge this gap,
we propose DeepResonance, a multimodal music understanding LLM fine-tuned via
multi-way instruction tuning with multi-way aligned music, text, image, and
video data. To this end, we construct Music4way-MI2T, Music4way-MV2T, and
Music4way-Any2T, three 4-way training and evaluation datasets designed to
enable DeepResonance to integrate both visual and textual music feature
content. We also introduce multi-sampled ImageBind embeddings and a
pre-alignment Transformer to enhance modality fusion prior to input into text
LLMs, tailoring DeepResonance for multi-way instruction tuning. Our model
achieves state-of-the-art performances across six music understanding tasks,
highlighting the benefits of the auxiliary modalities and the structural
superiority of DeepResonance. 

FUTGA: Towards Fine-grained Music Understanding through Temporally-enhanced Generative Augmentation
wu2024futga
2024
Existing music captioning methods are limited to generating concise global
descriptions of short music clips, which fail to capture fine-grained musical
characteristics and time-aware musical changes. To address these limitations,
we propose FUTGA, a model equipped with fined-grained music understanding
capabilities through learning from generative augmentation with temporal
compositions. We leverage existing music caption datasets and large language
models (LLMs) to synthesize fine-grained music captions with structural
descriptions and time boundaries for full-length songs. Augmented by the
proposed synthetic dataset, FUTGA is enabled to identify the music's temporal
changes at key transition points and their musical functions, as well as
generate detailed descriptions for each music segment. We further introduce a
full-length music caption dataset generated by FUTGA, as the augmentation of
the MusicCaps and the Song Describer datasets. We evaluate the automatically
generated captions on several downstream tasks, including music generation and
retrieval. The experiments demonstrate the quality of the generated captions
and the better performance in various downstream tasks achieved by the proposed
music captioning approach. 

Pengi: An Audio Language Model for Audio Tasks
deshmukh2023pengi
2023
In the domain of audio processing, Transfer Learning has facilitated the rise
of Self-Supervised Learning and Zero-Shot Learning techniques. These approaches
have led to the development of versatile models capable of tackling a wide
array of tasks, while delivering state-of-the-art performance. However, current
models inherently lack the capacity to produce the requisite language for
open-ended tasks, such as Audio Captioning or Audio Question & Answering. We
introduce Pengi, a novel Audio Language Model that leverages Transfer Learning
by framing all audio tasks as text-generation tasks. It takes as input, an
audio recording, and text, and generates free-form text as output. The input
audio is represented as a sequence of continuous embeddings by an audio
encoder. A text encoder does the same for the corresponding text input. Both
sequences are combined as a prefix to prompt a pre-trained frozen language
model. The unified architecture of Pengi enables open-ended tasks and
close-ended tasks without any additional fine-tuning or task-specific
extensions. When evaluated on 22 downstream tasks, our approach yields
state-of-the-art performance in several of them. Our results show that
connecting language models with audio models is a major step towards
general-purpose audio understanding

SALMONN: Towards Generic Hearing Abilities for Large Language Models
tang2024salmonn
2023
Hearing is arguably an essential ability of artificial intelligence (AI) agents in the physical world, which refers to the perception and understanding of general auditory information consisting of at least three types of sounds: speech, audio events, and music. In this paper, we propose SALMONN, a speech audio language music open neural network, built by integrating a pre-trained text-based large language model (LLM) with speech and audio encoders into a single multimodal model. SALMONN enables the LLM to directly process and understand general audio inputs and achieve competitive performances on a number of speech and audio tasks used in training, such as automatic speech recognition and translation, auditory-information-based question answering, emotion recognition, speaker verification, and music and audio captioning etc. SALMONN also has a diverse set of emergent abilities unseen in the training, which includes but is not limited to speech translation to untrained languages, speech-based slot filling, spoken-query-based question answering, audio-based storytelling, and speech audio co-reasoning etc. The presence of cross-modal emergent abilities is studied, and a novel few-shot activation tuning approach is proposed to activate such abilities. To our knowledge, SALMONN is the first model of its type and can be regarded as a step towards AI with generic hearing abilities.

Qwen-Audio: Advancing Universal Audio Understanding via Unified Large-Scale Audio-Language Models
chu2023qwen
2023
Recently, instruction-following audio-language models have received broad attention for audio interaction with humans. However, the absence of pre-trained audio models capable of handling diverse audio types and tasks has hindered progress in this field. Consequently, most existing works have only been able to support a limited range of interaction capabilities. In this paper, we develop the Qwen-Audio model and address this limitation by scaling up audio-language pre-training to cover over 30 tasks and various audio types, such as human speech, natural sounds, music, and songs, to facilitate universal audio understanding abilities. However, directly co-training all tasks and datasets can lead to interference issues, as the textual labels associated with different datasets exhibit considerable variations due to differences in task focus, language, granularity of annotation, and text structure. To overcome the one-to-many interference, we carefully design a multi-task training framework by conditioning on a sequence of hierarchical tags to the decoder for encouraging knowledge sharing and avoiding interference through shared and specified tags respectively. Remarkably, Qwen-Audio achieves impressive performance across diverse benchmark tasks without requiring any task-specific fine-tuning, surpassing its counterparts. Building upon the capabilities of Qwen-Audio, we further develop Qwen-Audio-Chat, which allows for input from various audios and text inputs, enabling multi-turn dialogues and supporting various audio-central scenarios. 

M2UGen: Multi-modal Music Understanding and Generation with the Power of Large Language Models
liu2023m2ugen
2024
The current landscape of research leveraging large language models (LLMs) is experiencing a surge. Many works harness the powerful reasoning capabilities of these models to comprehend various modalities, such as text, speech, images, videos, etc. They also utilize LLMs to understand human intention and generate desired outputs like images, videos, and music. However, research that combines both understanding and generation using LLMs is still limited and in its nascent stage. To address this gap, we introduce a Multi-modal Music Understanding and Generation (M2UGen) framework that integrates LLM's abilities to comprehend and generate music for different modalities. The M2UGen framework is purpose-built to unlock creative potential from diverse sources of inspiration, encompassing music, image, and video through the use of pretrained MERT, ViT, and ViViT models, respectively. To enable music generation, we explore the use of AudioLDM 2 and MusicGen. Bridging multi-modal understanding and music generation is accomplished through the integration of the LLaMA 2 model. Furthermore, we make use of the MU-LLaMA model to generate extensive datasets that support text/image/video-to-music generation, facilitating the training of our M2UGen framework. We conduct a thorough evaluation of our proposed framework. The experimental results demonstrate that our model achieves or surpasses the performance of the current state-of-the-art models. 

A Survey of Foundation Models for Music Understanding
li2024survey
2024
Music is essential in daily life, fulfilling emotional and entertainment
needs, and connecting us personally, socially, and culturally. A better
understanding of music can enhance our emotions, cognitive skills, and cultural
connections. The rapid advancement of artificial intelligence (AI) has
introduced new ways to analyze music, aiming to replicate human understanding
of music and provide related services. While the traditional models focused on
audio features and simple tasks, the recent development of large language
models (LLMs) and foundation models (FMs), which excel in various fields by
integrating semantic information and demonstrating strong reasoning abilities,
could capture complex musical features and patterns, integrate music with
language and incorporate rich musical, emotional and psychological knowledge.
Therefore, they have the potential in handling complex music understanding
tasks from a semantic perspective, producing outputs closer to human
perception. This work, to our best knowledge, is one of the early reviews of
the intersection of AI techniques and music understanding. We investigated,
analyzed, and tested recent large-scale music foundation models in respect of
their music comprehension abilities. We also discussed their limitations and
proposed possible future directions, offering insights for researchers in this
field.
<comment>I used this during lit review.</comment>

Audio Flamingo 2: An Audio-Language Model with Long-Audio Understanding and Expert Reasoning Abilities
ghosh2025audio
2025
Understanding and reasoning over non-speech sounds and music are crucial for
both humans and AI agents to interact effectively with their environments. In
this paper, we introduce Audio Flamingo 2 (AF2), an Audio-Language Model (ALM)
with advanced audio understanding and reasoning capabilities. AF2 leverages (i)
a custom CLAP model, (ii) synthetic Audio QA data for fine-grained audio
reasoning, and (iii) a multi-stage curriculum learning strategy. AF2 achieves
state-of-the-art performance with only a 3B parameter small language model,
surpassing large open-source and proprietary models across over 20 benchmarks.
Next, for the first time, we extend audio understanding to long audio segments
(30 secs to 5 mins) and propose LongAudio, a large and novel dataset for
training ALMs on long audio captioning and question-answering tasks.
Fine-tuning AF2 on LongAudio leads to exceptional performance on our proposed
LongAudioBench, an expert annotated benchmark for evaluating ALMs on long audio
understanding capabilities. We conduct extensive ablation studies to confirm
the efficacy of our approach.

Solla: Towards a Speech-Oriented LLM That Hears Acoustic Context
ao2025solla
2025
Large Language Models (LLMs) have recently shown remarkable ability to
process not only text but also multimodal inputs such as speech and audio.
However, most existing models primarily focus on analyzing input signals using
text instructions, overlooking scenarios in which speech instructions and audio
are mixed and serve as inputs to the model. To address these challenges, we
introduce Solla, a novel framework designed to understand speech-based
questions and hear the acoustic context concurrently. Solla incorporates an
audio tagging module to effectively identify and represent audio events, as
well as an ASR-assisted prediction method to improve comprehension of spoken
content. To rigorously evaluate Solla and other publicly available models, we
propose a new benchmark dataset called SA-Eval, which includes three tasks:
audio event classification, audio captioning, and audio question answering.
SA-Eval has diverse speech instruction with various speaking styles,
encompassing two difficulty levels, easy and hard, to capture the range of
real-world acoustic conditions. Experimental results show that Solla performs
on par with or outperforms baseline models on both the easy and hard test sets,
underscoring its effectiveness in jointly understanding speech and audio.
<comment>
"In contrast to [most existing works], SOLLA distinguishes itself by emphasizing both the understanding of audio content and the processing of speech instructions"
Method-wise, it looks like just an input adaptor to the text LLM plus some audio tagging domain knowledge.
</comment>
'''.strip()

papers['grounding'] = r'''
Kosmos-2: Grounding Multimodal Large Language Models to the World
kosmos2
2023
We introduce Kosmos-2, a Multimodal Large Language Model (MLLM), enabling new capabilities of perceiving object descriptions (e.g., bounding boxes) and grounding text to the visual world. Specifically, we represent refer expressions as links in Markdown, i.e., ``[text span](bounding boxes)'', where object descriptions are sequences of location tokens. Together with multimodal corpora, we construct large-scale data of grounded image-text pairs (called GrIT) to train the model. In addition to the existing capabilities of MLLMs (e.g., perceiving general modalities, following instructions, and performing in-context learning), Kosmos-2 integrates the grounding capability into downstream applications. We evaluate Kosmos-2 on a wide range of tasks, including (i) multimodal grounding, such as referring expression comprehension, and phrase grounding, (ii) multimodal referring, such as referring expression generation, (iii) perception-language tasks, and (iv) language understanding and generation. This work lays out the foundation for the development of Embodiment AI and sheds light on the big convergence of language, multimodal perception, action, and world modeling, which is a key step toward artificial general intelligence.

Visual Position Prompt for MLLM based Visual Grounding
tang2025visual
2025
Although Multimodal Large Language Models (MLLMs) excel at various
image-related tasks, they encounter challenges in precisely aligning
coordinates with spatial information within images, particularly in
position-aware tasks such as visual grounding. This limitation arises from two
key factors. First, MLLMs lack explicit spatial references, making it difficult
to associate textual descriptions with precise image locations. Second, their
feature extraction processes prioritize global context over fine-grained
spatial details, leading to weak localization capability. To address this
issue, we introduce VPP-LLaVA, an MLLM equipped with Visual Position Prompt
(VPP) to improve its grounding capability. VPP-LLaVA integrates two
complementary mechanisms. The global VPP overlays learnable, axis-like
embeddings onto the input image to provide structured spatial cues. The local
VPP focuses on fine-grained localization by incorporating position-aware
queries, which suggests probable object locations. We also introduce a VPP-SFT
dataset with 0.6M samples, consolidating high-quality visual grounding data
into a compact format for efficient model training. Training on this dataset
with VPP enhances the model's performance, achieving state-of-the-art results
on standard grounding benchmarks despite using fewer training samples compared
to other MLLMs like MiniGPT-v2, which rely on much larger datasets ($\sim$21M
samples).

Large-scale Pre-training for Grounded Video Caption Generation
kazakos2025large
2025
We propose a novel approach for captioning and object grounding in video,
where the objects in the caption are grounded in the video via temporally dense
bounding boxes. We introduce the following contributions. First, we present a
large-scale automatic annotation method that aggregates captions grounded with
bounding boxes across individual frames into temporally dense and consistent
bounding box annotations. We apply this approach on the HowTo100M dataset to
construct a large-scale pre-training dataset, named HowToGround1M. We also
introduce a Grounded Video Caption Generation model, dubbed GROVE, and
pre-train the model on HowToGround1M. Second, we introduce a new dataset,
called iGround, of 3500 videos with manually annotated captions and dense
spatio-temporally grounded bounding boxes. This allows us to measure progress
on this challenging problem, as well as to fine-tune our model on this
small-scale but high-quality data. Third, we demonstrate that our approach
achieves state-of-the-art results on the proposed iGround dataset compared to a
number of baselines, as well as on the VidSTG and ActivityNet-Entities
datasets. We perform extensive ablations that demonstrate the importance of
pre-training using our automatically annotated HowToGround1M dataset followed
by fine-tuning on the manually annotated iGround dataset and validate the key
technical contributions of our model.

Omni-RGPT: Unifying Image and Video Region-level Understanding via Token Marks
heo2025omni
2025
We present Omni-RGPT, a multimodal large language model designed to
facilitate region-level comprehension for both images and videos. To achieve
consistent region representation across spatio-temporal dimensions, we
introduce Token Mark, a set of tokens highlighting the target regions within
the visual feature space. These tokens are directly embedded into spatial
regions using region prompts (e.g., boxes or masks) and simultaneously
incorporated into the text prompt to specify the target, establishing a direct
connection between visual and text tokens. To further support robust video
understanding without requiring tracklets, we introduce an auxiliary task that
guides Token Mark by leveraging the consistency of the tokens, enabling stable
region interpretation across the video. Additionally, we introduce a
large-scale region-level video instruction dataset (RegVID-300k). Omni-RGPT
achieves state-of-the-art results on image and video-based commonsense
reasoning benchmarks while showing strong performance in captioning and
referring expression comprehension tasks.

Ferret: Refer and Ground Anything Anywhere at Any Granularity
you2024ferret
2023
We introduce Ferret, a new Multimodal Large Language Model (MLLM) capable of
understanding spatial referring of any shape or granularity within an image and
accurately grounding open-vocabulary descriptions. To unify referring and
grounding in the LLM paradigm, Ferret employs a novel and powerful hybrid
region representation that integrates discrete coordinates and continuous
features jointly to represent a region in the image. To extract the continuous
features of versatile regions, we propose a spatial-aware visual sampler, adept
at handling varying sparsity across different shapes. Consequently, Ferret can
accept diverse region inputs, such as points, bounding boxes, and free-form
shapes. To bolster the desired capability of Ferret, we curate GRIT, a
comprehensive refer-and-ground instruction tuning dataset including 1.1M
samples that contain rich hierarchical spatial knowledge, with 95K hard
negative data to promote model robustness. The resulting model not only
achieves superior performance in classical referring and grounding tasks, but
also greatly outperforms existing MLLMs in region-based and
localization-demanded multimodal chatting. Our evaluations also reveal a
significantly improved capability of describing image details and a remarkable
alleviation in object hallucination.

Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond
bai2023qwen
2023
In this work, we introduce the Qwen-VL series, a set of large-scale
vision-language models (LVLMs) designed to perceive and understand both texts
and images. Starting from the Qwen-LM as a foundation, we endow it with visual
capacity by the meticulously designed (i) visual receptor, (ii) input-output
interface, (iii) 3-stage training pipeline, and (iv) multilingual multimodal
cleaned corpus. Beyond the conventional image description and
question-answering, we implement the grounding and text-reading ability of
Qwen-VLs by aligning image-caption-box tuples. The resulting models, including
Qwen-VL and Qwen-VL-Chat, set new records for generalist models under similar
model scales on a broad range of visual-centric benchmarks (e.g., image
captioning, question answering, visual grounding) and different settings (e.g.,
zero-shot, few-shot). Moreover, on real-world dialog benchmarks, our
instruction-tuned Qwen-VL-Chat also demonstrates superiority compared to
existing vision-language chatbots.
'''.strip()

STEP_BY_STEP_GUIDE = (
'''
Follow this plan:
1. Infer the timeline of mentioned papers. Mark anything that's potentially parallel.
2. State some common methodologies.
3. For each paper,
3.1. State its unique idea(s), if you can already identify.'''
# '''
# 3.2. State its relation to my paper.'''
'''
3.2. Note any important info you still need for the purpose of writing the Related Work paragraph. If nothing is critically missing, skip.
4. From the above, ask me three clarifying questions that critically help you with the literature review.
5. I will answer.
6. Revise the timeline.
7. Write the paragraph.
8. Idenitfy any assumptions you made beyond the factual information I gave you.
9. Check your assumptions by searching the internet.

In your next response, do steps 1 - 4.
'''
).strip() # use only for non-reasoning GPT models

format_ = '''
Citation style: write "~\\cite{insert_bib_hanle_here}" at the end of your sentence.
Enclose your paragraph in a code block ("```latex").
'''.strip()

def prompt(stage: str):
    return f'''
Write a paragraph in the Related Work Section in my academic paper in music AI.

Here is my paper:
<my_paper>
{ours[0]}
{ours[1]}
</my_paper>

{dict(
    understanding = 'In this paragraph, focus on music understanding.', 
    grounding = 'In this paragraph, focus on LLM grounding. The last sentence should situate my work in the literature.', 
)[stage]}

{format_}

Here are related papers you should integrate, containing title, bib_hanle, year, abstract (potentially multi-line), and optionally my comment:
<related_papers>
{papers[stage]}
</related_papers>
'''.strip()

for stage in STAGES:
    with open(f'prompt-{stage}.txt', 'w', encoding='utf-8') as f:
        f.write(prompt(stage))
