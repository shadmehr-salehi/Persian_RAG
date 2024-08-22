import os
import numpy as np
import faiss
import ocrmypdf
from transformers import AutoTokenizer, AutoModelForCausalLM  ,BitsAndBytesConfig ,TextStreamer , GenerationConfig
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# from huggingface_hub import login
# from huggingface_hub import hf_hub_download
# import bitsandbytes as bnb
# import fasttext
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# import logging
# ocrmypdf.configure_logging(verbosity=-1)
# login(token=os.getenv('HUGGINGFACE_HUB_TOKEN'))
# import warnings
# warnings.filterwarnings("ignore")

class PersianRAG:
    def __init__(self) -> None:
        bconf = BitsAndBytesConfig(
            load_in_8bit=True)
        
        model_path = "/app/model"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path , 
        device_map={"": "cuda"} , 
        quantization_config=bconf                                               
        )
        # self.streamer = TextStreamer(self.tokenizer,  skip_prompt=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map={"": "cuda"} , 
            quantization_config=bconf
        )
        self.embedding_m = HuggingFaceEmbeddings(
            model_name='intfloat/multilingual-e5-large'
        )
        self.indices = {}
        self.chunks = {} 
        self.languages = {}
        # embedding_model_path = hf_hub_download(repo_id="facebook/fasttext-fa-vectors", filename="model.bin")
        
        # self.embedding_model = fasttext.load_model(embedding_model_path)
        


    def set_language(self,language:str , user_id:str) -> None:
        # self.language = language
        PROMPT_TEMPLATE = \
        f"""
        Use the following pieces of information to answer the user's question,
        if they mention specific language, answer in that language, else answer with language of question.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.  
        ---
        context:
        {{context}}
        ---
        answer the following question based only on the information provided above.
        
        Question : 
        {{question}} 
        -------
        Only return the helpful answer below and nothing else.
        Helpful answer:
        """
        self.prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        # language_coded = {
        #     "Persian": "fa",
        #     "English": "en",
        #     "French" : "fr",
        #     "Japanese" : "ja",
        # }
        # embedding_model_path = f"/app/model/embd/{language}/model.bin"
        # self.embedding_model = fasttext.load_model(embedding_model_path)
        self.languages[user_id] = language
        
        
    def parse_file(self,filepath:str,user_id: str,chunk_size=800,chunk_overlap=100) -> None:
        
        # Extract text from the file
        if filepath.endswith(".pdf"):
            lang_mapping = {
                "persian": "fas",
                "english": "eng",
                "french": "fra",
                "japanese": "jpn"
            }
            
            ocrmypdf.ocr(filepath, f"-", 
                         deskew=False,
                         output_type='none' , 
                         force_ocr=True ,
                         sidecar=f"{filepath.rstrip('.pdf')}.txt" ,
                         language=lang_mapping[self.languages[user_id]] ,
                         progress_bar=True)
            with open(f"{filepath.rstrip('.pdf')}.txt", 'r') as file:
                text = file.read()
                
        elif filepath.endswith(".txt"):
            with open(filepath, 'r') as file:
                text = file.read()
                
        text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
        )
        chunks = text_splitter.split_text(text)

        # Preprocess each chunk
        preprocessed_chunks = [chunk.replace('\n', ' ').strip() for chunk in chunks]
        embedded_chunks = [self.embedding_m.embed_query(f"query: {chunk} ") for chunk in preprocessed_chunks]
        # embedded_chunks = [self.embedding_model.get_sentence_vector(chunk) for chunk in preprocessed_chunks]
        embedded_chunks_np = np.array(embedded_chunks).astype('float32')
        embedded_chunks_np = embedded_chunks_np.reshape(-1, embedded_chunks_np.shape[-1])
        
        # Create a FAISS index
        dim = embedded_chunks_np.shape[1]
        index = faiss.IndexFlatL2(dim )
        index.add(embedded_chunks_np)
        # print("chunks shape",embedded_chunks_np.shape)
        
        self.indices[user_id] = index
        self.chunks[user_id] = chunks
        
    def query(self,query:str,user_id: str,min_tokens=100,max_tokens=1500,temp=0.2) -> str:
        if user_id not in self.indices or user_id not in self.chunks:
            return "User context not found. Please upload a file first."
        
        
        query_embedding = np.array(self.embedding_m.embed_query(query)).reshape(1, -1).astype('float32')
        # print("query shape",query_embedding.shape)
        distances, indices = self.indices[user_id].search(query_embedding, 7)
        # print(f"dists: {distances} \n -------- \n indices: {indices}")
        context_text = "\n\n---\n\n".join([self.chunks[user_id][i] for i in indices[0]])
        prompt = self.prompt_template.format(context=context_text, question=query)
        messages = [
        # {"role": "system", "content": "you"},
        {"role": "user", "content": f"{prompt}"}
        ]
        input_ids = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
        input_ids = input_ids.to('cuda')
        gn = GenerationConfig(
            min_new_tokens=min_tokens,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temp,
            repitition_penalty=1.7,
            # streamer = self.streamer
        )
        gen_tokens = self.model.generate(
            input_ids,
            generation_config = gn
            )

        gen_text = self.tokenizer.decode(gen_tokens[0][len(input_ids[0]):], skip_special_tokens=True)
        return gen_text
    
class QueryRequest(BaseModel):
    query: str
    user_id: str

class FileRequest(BaseModel):
    filepath: str
    user_id: str

class LangRequest(BaseModel):
    lang: str
    user_id: str
    
app = FastAPI()
rag_model = PersianRAG()

@app.post("/query")
def query_rag(request: QueryRequest):
    try:
        response = rag_model.query(request.query, user_id=str(request.user_id))
        return {"response": response}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/parse-file")
def parse_file(file_request: FileRequest):
    try:
        rag_model.parse_file(file_request.filepath, user_id=str(file_request.user_id))
        return {"response": "File parsed successfully."}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/set-language")
def set_language(language: LangRequest):
    try:
        rag_model.set_language(language.lang , user_id=str(language.user_id) )
        return {"response": "Language set successfully."}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

print("Model loaded successfully.")
# print(rag_model.languages)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)