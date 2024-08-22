import chainlit as cl
import requests
from chainlit.input_widget import Select

@cl.set_chat_profiles
async def chat_profile():
    return [
        cl.ChatProfile(
            name="Persian",
            markdown_description="For Parsing Persian language.",
            # icon="https://picsum.photos/200",
        ),
        cl.ChatProfile(
            name="English",
            markdown_description="For Parsing English language.",
            # icon="https://picsum.photos/250",
        ),
        cl.ChatProfile(
            name="French",
            markdown_description="For Parsing French language.",
            # icon="https://picsum.photos/250",
        ),
        cl.ChatProfile(
            name="Japanese",
            markdown_description="For Parsing Japanese language.",
            # icon="https://picsum.photos/250",
        ),
    ]

@cl.on_chat_start
async def on_chat_start():
    # ask the user to input the language
    # settings = await cl.ChatSettings(
    #     [
    #         Select(
    #             id="Model",
    #             label="OpenAI - Model",
    #             values=["gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4", "gpt-4-32k"],
    #             initial_index=0,
    #         )
    #     ]
    # ).send()
    # value = settings["Model"]
    # msg = cl.Message(content=f"Processing `{value}`...", disable_feedback=True)
    # await msg.send()
    uid = cl.user_session.get("id")

    language = cl.user_session.get("chat_profile")
    await cl.Message(
        content=f"starting chat using the {language} language {uid}" ,
        disable_feedback=True
    ).send()
    print({"lang": language.lower() , "user_id": uid})
    response = requests.post("http://ayaengine:5000/set-language", json={"lang": language.lower() , "user_id": uid})
    if response.status_code == 200:
        response_data = response.json()
        await cl.Message(content=response_data["response"]).send()
    else:
        await cl.Message(content="Failed to get a response. Please try again!").send()
    
    files = None
    # Wait for the user to upload a file
    while files == None:
        files = await cl.AskFileMessage(
            content=f"Please upload a text or pdf file to begin! ",
            accept=["text/plain", "application/pdf"],
            max_size_mb=20,
            timeout=280,
        ).send()

    file = files[0]
    msg = cl.Message(content=f"Processing `{file.name}`...", disable_feedback=True )
    await msg.send()
    def req():
        return requests.post("http://ayaengine:5000/parse-file", json={"filepath": file.path , "user_id": uid} , timeout=600)
    response = await cl.make_async(req)()
    # print(response)
    if response.status_code == 200:
        response_data = response.json()
        # print("Response:", response_data["response"])
        msg.content = f"Processing `{file.name}` done. You can now ask questions!"
    await msg.update()
    
    
@cl.on_message
async def main(message: cl.Message):
    uid = cl.user_session.get("id")
    def answer():
        return requests.post("http://ayaengine:5000/query", json={"query": message.content , "user_id": uid} ,  timeout=600)
    response = await cl.make_async(answer)()
    if response.status_code == 200:
        response_data = response.json()
        await cl.Message(content=response_data["response"]).send()
    else:
        await cl.Message(content="Failed to get a response. Please try again!").send()