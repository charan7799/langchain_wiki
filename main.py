## Integrate our code to openai-API
import os
from constants import openai_key, google_custom_key, google_cx
# import langchain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain.memory import ConversationBufferMemory
import requests
from PIL import Image
from io import BytesIO

import gradio as gr
import streamlit as st

os.environ["OPENAI_API_KEY"]=openai_key

# streamlit framework
# os.
# st.title("WIKI Anime Character")
# input_text = st.text_input("search a Anime Character")

class googleImageSearch():
    def __init__(self, api_key, cx, num=5):
        
        self.api_key = api_key,
        self.cx = cx,
        self.num_examples=num

    

    def fetch_google_images(self, query, series_title):
        base_url = "https://www.googleapis.com/customsearch/v1"
        
        if series_title:
            query = "Anime Character "+ query+" in Anime Series "+series_title

        params = {
            'key': self.api_key,
            'cx': self.cx,
            'q': query,
            'searchType': 'image',
            'num': self.num_examples
        }

        response = requests.get(base_url, params=params)
        data = response.json()

        if 'items' in data:
            item_list =  [item['link'] for item in data['items']]
            self.imgLinksList = item_list
        else:
            print("Error: ", data)
            self.imgLinksList =  []

        self.output = []
        for item in self.imgLinksList:
            response = requests.get(item)
            
            if response.status_code == 200:
                self.output.append(Image.open(BytesIO(response.content)))
            else:
                print(f"Failed to download image from {item}. Status code: {response.status_code}")
                return self.output
        return self.output

def LLMchain_wiki(text_prompt):
    # memory
    character_name_memory = ConversationBufferMemory(input_key='name', memory_key='name_character')
    Anime_memory = ConversationBufferMemory(input_key='name', memory_key='Anime_name')
    # Appearance_memory = ConversationBufferMemory(input_key=['name', 'Anime'], memory_key='Appearance_history')
    descr_memory = ConversationBufferMemory(input_key='Anime', memory_key='plot_summary_history')

    # Prompt Templates

    # first_input_prompt=PromptTemplate(
    #     input_variables=['name'],
    #     template="Tell me about Anime Character named {name}"
    # )

    ## OPENAI LLMS
    llm = OpenAI(temperature=0.4)

    # chain = LLMChain(llm=llm, prompt=first_input_prompt, verbose=True, output_key='character_info', memory=character_name_memory)
    # doc = nlp(output_text)

    # Extract character_name names
    # character_name_names = [ent.text for ent in doc.ents if ent.label_ == "character_name"]
    first_input_prompt=PromptTemplate(
        input_variables=['name'],
        template="give the Title of the Anime, In which {name} appeared in the mentioned template in triple quotes\
            '''\
            Title:    \
            Date released: \
            ''' \  "
    )

    chain2 = LLMChain(llm=llm, prompt=first_input_prompt, verbose=False, output_key='Anime', memory=Anime_memory)

    second_input_prompt=PromptTemplate(
        input_variables=['name', 'Anime'],
        template="name any 5 of the initial and top Appearances of {name} in {Anime} in order"
    )

    chainTemp = LLMChain(llm=llm, prompt=second_input_prompt, verbose=True, output_key='Appearances')

    third_input_prompt=PromptTemplate(
        input_variables=['name', 'Anime'],
        template="summarize the role of {name} in Anime {Anime}"
    )
    chain3=LLMChain(llm=llm,prompt=third_input_prompt,verbose=True,output_key='plot_summary', memory=descr_memory)

    parent_chain = SequentialChain(chains=[chain2, chainTemp, chain3], input_variables=['name'], output_variables=['Anime', 'Appearances', 'plot_summary'], verbose=True)
    result = parent_chain({'name': text_prompt})
    

    # result = get_completion(base64_image)
    return result['Anime'], result['Appearances'], result['plot_summary']

if __name__ == '__main__':
    # st.write(parent_chain({'name': input_text}))

    # with st.expander('Character Description'): 
    #     st.info(Anime_memory.buffer)

    # with st.expander('Plot summary'): 
    #     st.info(descr_memory.buffer)
    gr.close_all()
    # demo = gr.Interface(fn=LLMchain_wiki,
    #                     inputs=[gr.Textbox(label="Search about an anime character", lines=1)],
    #                     outputs=[gr.Textbox(label="Character info", lines=10), gr.Textbox(label="Anime Appearance", lines=3), \
    #                              gr.Textbox(label="Anime plot summary", lines=10), gr.Image(label="Image")],
    #                     title="ANIME WIKI",
    #                     description="Search about an anime character",
    #                     allow_flagging="never",
    #                     theme=gr.themes.Base(),
    #                     )
    demo = gr.Blocks(theme=gr.themes.Soft())

    with demo:
        with gr.Row():
            name_query = gr.Textbox(label="Search about an anime character", lines=1)
        # text = gr.Textbox()
        # label = gr.Label()
        gImageSearch = googleImageSearch(api_key=google_custom_key, cx=google_cx)
        # gImageSearch.fetch_google_images(name_query)
        # outputImages = gImageSearch.get_queryImages()
        with gr.Row():
            with gr.Column():
                btn = gr.Button("Generate images")
                gallery = gr.Gallery(
                    label="Generated images", show_label=False, elem_id="gallery"
                    , container=True, columns=[3], rows=[2], object_fit="cover", height=240, show_download_button=True, preview=False)
                chara_info = gr.Textbox(label="Anime", lines=1)

            with gr.Column():
                b1 = gr.Button("get info")
                plot_info = gr.Textbox(label="Role in Anime", lines=5)
                appearance_info = gr.Textbox(label="Anime Appearance", lines=5)
        # b2 = gr.Button("get image")
        
        b1.click(LLMchain_wiki, inputs=name_query, outputs=[chara_info, appearance_info, plot_info])
        if chara_info:
            # name_query = 
            btn.click(gImageSearch.fetch_google_images, [name_query, chara_info], gallery)
        else:
            btn.click(gImageSearch.fetch_google_images, [name_query, None], gallery)
        # b2.click(fn=None, inputs=[], outputs=gr.Dropdown(choices=outputImages))

    demo.launch(share=True)