
import pandas as pd
import os
import requests
import json
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
import gradio as gr
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load the translation model and tokenizer
print('Loading translation model and tokenizer...')
translation_model_name = "facebook/nllb-200-distilled-600M"
translation_model = AutoModelForSeq2SeqLM.from_pretrained(
    translation_model_name)
translation_tokenizer = AutoTokenizer.from_pretrained(translation_model_name)

# Instantiate the translation pipelines
deu_translator = pipeline('translation', model=translation_model,
                          tokenizer=translation_tokenizer, src_lang="en", tgt_lang="de")
spa_translator = pipeline('translation', model=translation_model,
                          tokenizer=translation_tokenizer, src_lang="en", tgt_lang="es")
fra_translator = pipeline('translation', model=translation_model,
                          tokenizer=translation_tokenizer, src_lang="en", tgt_lang="fr")

# Load the model and tokenizer
print('Loading QA model and tokenizer...')
model_name = "deepset/electra-base-squad2"
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

qa_pipeline = pipeline('question-answering',
                       model=model_name, tokenizer=model_name)


# Set up MACULA data as pandas dataframe
print('Downloading MACULA Greek data...')


def download_file(url, file_name):
    response = requests.get(url)
    with open(file_name, 'wb') as file:
        file.write(response.content)


file1_url = 'https://raw.githubusercontent.com/Clear-Bible/macula-greek/main/Nestle1904/TSV/macula-greek.tsv'
file2_url = 'https://raw.githubusercontent.com/Clear-Bible/macula-greek/main/sources/MARBLE/SDBG/marble-domain-label-mapping.json'
file1_name = 'macula-greek.tsv'
file2_name = 'marble-domain-label-mapping.json'

if file1_name not in os.listdir():
    download_file(file1_url, file1_name)

if file2_name not in os.listdir():
    download_file(file2_url, file2_name)

# Import Macula Greek data
mg = pd.read_csv('macula-greek.tsv', index_col='xml:id', sep='\t',
                 header=0, converters={'*': str}).fillna('missing')
# mg['domain'] = mg['domain'].astype(str).fillna('missing')

# Extract book, chapter, and verse into separate columns
mg[['book', 'chapter', 'verse']] = mg['ref'].str.extract(
    r'(\d?[A-Z]+)\s(\d+):(\d+)')

# Add columns for book + chapter, and book + chapter + verse for easier grouping
mg['book_chapter'] = mg['book'] + ' ' + mg['chapter'].astype(str)
mg['book_chapter_verse'] = mg['book_chapter'] + ':' + mg['verse'].astype(str)

# Import domain-label mapping

# Open the JSON file
with open('marble-domain-label-mapping.json', 'r') as f:

    # Load the contents of the file as a dictionary
    domain_labels = json.load(f)

domain_labels['missing'] = 'no domain'
domain_labels['nan'] = 'no domain'

# Use domain labels to create a new column


def get_domain_label(domain_string_number):
    labels = [domain_labels[label]
              for label in domain_string_number.split(' ')]
    return labels


mg['domain_label'] = mg['domain'].apply(get_domain_label)

# to get data for a specific word, use the following code:
# mg.loc['n40001001002'].to_dict() where n40001001002 is the word ID

# Create a dictionary with attribute descriptions
attribute_descriptions = {
    "after": "Encodes the following character, including a blank space.",
    "articular": "'true' if the word has an article (i.e., modified by the word 'the').",
    "case": "Grammatical case: nominative, genitive, dative, accusative, or vocative",
    "class": "On words, the class is the word's part of speech",
    "cltype": "Explicitly marks Verbless Clauses, Verb Elided Clauses, and Minor Clauses",
    "degree": "A derivative lexical category, indicating the degree of the adjective",
    "discontinuous": "'true' if the word is discontinuous with respect to sentence order due to reordering in the syntax tree",
    "domain": "Semantic domain information from the Semantic Dictionary of Biblical Greek (SDBG)",
    "frame": "Frames of verbs, refers to the arguments of the verb",
    "gender": "Grammatical gender values",
    "gloss": "SIL data, not Berean",
    "lemma": "Form of the word as it appears in a dictionary.",
    "ln": "Short for Louw-Nida, representing the semantic domain entry in Johannes P. Louw and Eugene Albert Nida, Greek-English Lexicon of the New Testament: Based on Semantic Domains (New York: United Bible Societies, 1996).",
    "mood": "Grammatical mood",
    "morph": "Morphological parsing codes",
    "normalized": "The normalized form of the token (i.e., no trailing or leading punctuation or accent shifting depending on context)",
    "number": "Grammatical number",
    "person": "Grammatical person",
    "ref": "Verse!word reference to this edition of the Nestle1904 text by USFM id",
    "referent": "The xml:id of the node to which a pronoun (i.e., 'he') refers. Note that some of these IDs are not word IDs but rather phrase or clause IDs.",
    "role": "The clause-level role of the word.",
    "strong": "Strong's number for the lemma",
    "subjref": "The xml:id of the node that is the implied subject of a verb (for verbs without an explicit subject). Note that some of these IDs are not word IDs but rather phrase or clause IDs.",
    "tense": "Grammatical tense form",
    "text": "Text content associated with the ID",
    "type": "Indicates different types of pronominals",
    "voice": "Grammatical voice",
    "xml:id": "XML ids occur on every word and encode the corpus ('n' for New Testament), the book (40 for Matthew), the chapter (001), verse (001), and word (001)."
}


def generate_prosaic_context(word_id, selected_fields=None):
    word_data = mg.loc[word_id].to_dict()
    prompt = f"{word_data['lemma']}'s "

    if not selected_fields:
        selected_fields = list(attribute_descriptions.keys())

    descriptions = []
    for key in selected_fields:
        value = word_data.get(key)
        if value not in (None, 'missing', 'nan'):
            attribute_description = attribute_descriptions.get(key)
            descriptions.append(f"{attribute_description} ({key}): {value}")

    prompt += ", ".join(descriptions)
    return prompt

# to generate prosaic context: prosaic_context = generate_prosaic_context(word_id)


# to return a sentence based on the book_chapter_verse value:
# mg[mg['book_chapter_verse'] == 'MAT 1:1'] # this will return every row for this verse
# for row in mg[mg['book_chapter_verse'] == 'MAT 1:1'].itertuples():
#     print(row)

# create a set from mg['book_chapter_verse'].unique()
unique_book_chapter_verse = set(mg['book_chapter_verse'].unique())

verseRef = 'JHN 3:16'

# def extract_text_and_gloss(verseRef):
#     if verseRef not in unique_book_chapter_verse:
#         return {text: 'verse not found', gloss: ''}
#     result = {}
#     for _, row in mg[mg['book_chapter_verse'] == verseRef].iterrows():
#         result[row['text']] = row['gloss']
#     return result

# text_and_gloss = extract_text_and_gloss(verseRef) # to get text and gloss


# Define some helper functions

def get_contextual_data(tokenId):
    prosaic_context = generate_prosaic_context(tokenId)
    return prosaic_context


def get_verse_content(verseRef, dataframe):
    unique_book_chapter_verse = set(dataframe['book_chapter_verse'])
    if verseRef not in unique_book_chapter_verse:
        return []
    matching_rows = dataframe[dataframe['book_chapter_verse'] == verseRef]
    tokens = [{"id": idx, "text": row['text'], "gloss": row['gloss']}
              for idx, row in matching_rows.iterrows()]
    return tokens


def answer_question(question, context):
    input_dict = {'question': question, 'context': context}
    answer = qa_pipeline(input_dict)
    return answer['answer']


def translate_text(text):
    deu_result = deu_translator(text)[0]['translation_text']
    spa_result = spa_translator(text)[0]['translation_text']
    fra_result = fra_translator(text)[0]['translation_text']
    return deu_result, spa_result, fra_result


def gradio_get_verse_content(verseRef):
    tokens = get_verse_content(verseRef, mg)
    return tokens


def gradio_wrapper(inputs, context):
    question = inputs
    answer = answer_question(question, context)
    deu_translation, spa_translation, fra_translation = translate_text(answer)
    return answer, deu_translation, spa_translation, fra_translation


state = {"tokens": []}

with gr.Blocks() as block:
    verse_reference = gr.Textbox(lines=1, label="Verse Reference")
    get_verse_content_button.click(
        gradio_get_verse_content, inputs=verse_reference, outputs=state["tokens"])
    
    
    question = gr.Textbox(lines=2, label="Question")
    context = gr.components.HTML(
        '''<textarea id="context" readonly rows="5" cols="40" style="border-style: solid; padding: 16px; margin-bottom: 16px; font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;"></textarea>''', label="Context")

    answer_eng = gr.Textbox(label="Answer (English)")
    answer_deu = gr.Textbox(label="Translation (German)")
    answer_spa = gr.Textbox(label="Translation (Spanish)")
    answer_fra = gr.Textbox(label="Translation (French)")

    get_verse_content_button = gr.Button("Get Verse Content")
    get_context_button = gr.Button("Get Context")

    # Linking functions
    
    get_context_button.click(
        get_contextual_data, inputs=state["tokens"], outputs=context)

    # gradio_wrapper_link = gr.Link(gradio_wrapper, inputs=[question, context], outputs=[
    #                               answer_eng, answer_deu, answer_spa, answer_fra])

    block.launch()


# iface_get_verse_content = gr.Interface(
#     fn=gradio_get_verse_content,
#     inputs=gr.Textbox(lines=1, label="Verse Reference"),
#     outputs="json",
#     allow_flagging=False,
# )


# # Define custom HTML template
# custom_template = """
# {% extends "base.html" %}
# {% block input %}
#     {{ super() }}
#     <div id="verseContent" style="border-style: solid; padding: 16px; margin-bottom: 16px; font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;"></div>
#     <script>
#         // Add 'id' attributes to the input elements
#         document.querySelector('.input_text:nth-child(1)').id = 'verseRef';
#         document.querySelector('.input_text:nth-child(2)').id = 'question';
#         document.querySelector('.input_text:nth-child(3)').id = 'context';

#         // Set the 'readonly' attribute for the 'context' input element
#         document.querySelector('#context').readOnly = true;

#         document.querySelector('#verseRef').addEventListener('change', async function() {
#             const verseRef = this.value;
#             const tokens = await iface_get_verse_content.predict(verseRef);
#             const verseContentDiv = document.querySelector('#verseContent');
#             verseContentDiv.innerHTML = tokens.map(token => `
#                 <span style="cursor: pointer; margin-right: 5px;" onclick="handleTokenClick(${token.id})">${token.text} (${token.gloss})</span>
#             `).join('');
#         });

#         function handleTokenClick(tokenId) {
#             const get_contextual_data = iface.getComponent("get_contextual_data");
#             get_contextual_data(tokenId).then(context => {
#                 document.querySelector('#context').value = context;
#             });
#         }
#     </script>
# {% endblock %}
# """

# # Define the Gradio interface
# iface = gr.Interface(
#     fn=gradio_wrapper,
#     inputs=[
#         gr.Textbox(lines=1, label="Verse Reference"),
#         gr.Textbox(lines=2, label="Question"),
#         gr.components.HTML(
#             '''<textarea id="context" readonly rows="5" cols="40" style="border-style: solid; padding: 16px; margin-bottom: 16px; font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;"></textarea>''', label="Context"),
#     ],
#     outputs=[
#         gr.Textbox(label="Answer (English)"),
#         gr.Textbox(label="Translation (German)"),
#         gr.Textbox(label="Translation (Spanish)"),
#         gr.Textbox(label="Translation (French)"),
#     ],
#     allow_flagging=False,
#     config={"custom_template": custom_template}
# )

# iface.allow_duplicates = True

# # Start the Gradio app
# if __name__ == "__main__":
#     gr.launch_multiple([iface, iface_get_verse_content])
