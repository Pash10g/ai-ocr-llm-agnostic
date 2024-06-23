import streamlit as st
from pymongo import MongoClient
from bson.objectid import ObjectId
import openai
from PIL import Image
import io
import os
import base64
import google.generativeai as genai
import json
import boto3


# OpenAI API key
openai.api_key = os.environ.get("OPENAI_API_KEY")

# Gemini API key
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
global gemini_pro
gemini_pro = genai.GenerativeModel('gemini-pro')
## Bedrok

bedrock_runtime = boto3.client('bedrock-runtime',
                               aws_access_key_id=os.environ.get('AWS_ACCESS_KEY'),
                               aws_secret_access_key=os.environ.get('AWS_SECRET_KEY'),
                               region_name="us-east-1")




API_CODE = os.environ.get("API_CODE")
# MongoDB connection
client = MongoClient(os.environ.get("MONGODB_ATLAS_URI"))
db = client['ocr_db']
global collection
collection = db['openai_documents']
global model

### Dropdown to choose the model


auth_collection=db['api_keys']
# Initialize session state for authentication
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

if 'messages' not in st.session_state:
    st.session_state.messages = []


def auth_form():
    

    st.write("Please enter the API code to access the application.")
    api_code = st.text_input("API Code", type="password")
    if st.button("Submit"):
        st.toast("Authenticating...", icon="⚠️")
        db_api_key=auth_collection.find_one({"api_key":api_code})
        if db_api_key:
            st.session_state.authenticated = True
            st.session_state.api_code = api_code
            st.success("Authentication successful.")
            st.rerun()  # Re-run the script to remove the auth form
        else:
            st.error("Authentication failed. Please try again.")



transcribed_object = "other"

def generate_image_description_with_claude(image):
    claude_body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1000,
        "system": f"Please trunscribe this {transcribed_object} into a json only output for MongoDB store, calture all data as a single document. Always have a 'name', 'summary' (for embedding ) and 'type' top field (type is a subdocument with user and 'ai_classified') as well as other fields as you see fit.",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image}}
                 
            ]
        }]
    })

    claude_response = bedrock_runtime.invoke_model(
        body=claude_body,
        modelId="anthropic.claude-3-5-sonnet-20240620-v1:0",
        accept="application/json",
        contentType="application/json",
    )
    response_body = json.loads(claude_response.get("body").read())
    # Assuming the response contains a field 'content' with the description
    return f"""```json
    {response_body["content"][0]['text']}
    """

# Function to transform image to text using OpenAI
def transform_image_to_text(image, format,model):

  
       
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format=format)
    img_byte_arr = img_byte_arr.getvalue()
    encoded_image = base64.b64encode(img_byte_arr).decode('utf-8')
    if (model == "gpt-4o"):
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{
            "role": "system",
            "content": "You are an ocr to json expert looking to transcribe an image. If the type is 'other' please specify the type of object and clasiffy as you see fit."
            },
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": f"Please trunscribe this {transcribed_object} into a json only output for MongoDB store, calture all data as a single document. Always have a 'name', 'summary' (for embedding ) and 'type' top field (type is a subdocument with user and 'ai_classified') as well as other fields as you see fit."
            },
            {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{encoded_image}"
            }
            }
        ]
        }
    ]
        )
        extracted_text = response.choices[0].message.content
        return extracted_text
    elif(model == "gemini-pro"):
        vision = genai.GenerativeModel('gemini-pro-vision')
        response = vision.generate_content([f"You are an ocr to json expert looking to transcribe an image. If the type is 'other' please specify the type of object and clasiffy as you see fit. Please trunscribe this {transcribed_object} into a json only output for MongoDB store. Always have a 'name', 'summary' and 'type' top field (type is a subdocument with user and 'ai_classified') as well as other fields detailed as possible.", image], stream=False)
        return response.text
    elif(model == "bedrock-claude-3"):
        response = generate_image_description_with_claude(encoded_image)
        return response
    



def clean_document(document):
    cleaned_document = document.strip().strip("```json").strip("```").strip()
    return json.loads(cleaned_document)

def openai_embedding(text):
    response = openai.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def get_embedding_from_titan_multimodal(text):

    body = json.dumps({
        "inputText": text
    })
     
    bedrock_resp = bedrock_runtime.invoke_model(
        body=body,
        modelId="amazon.titan-embed-text-v2:0",
        accept="application/json",
        contentType="application/json",
    )
     
    response_body = json.loads(bedrock_resp.get("body").read())
    return response_body["embedding"]

def get_embedding_from_gemini(text):
    result = genai.embed_content(
        model="models/embedding-001",
        content=text,
        task_type="retrieval_document",
        title="Embedding of single string"
    )
    return result['embedding']

# Function to save image and text to MongoDB
def save_image_to_mongodb(image, description, embedding_model):
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format=image.format)
    img_byte_arr = img_byte_arr.getvalue()
    encoded_image = base64.b64encode(img_byte_arr).decode('utf-8')
    
    # Remove the ```json and ``` parts
    

    # Parse the cleaned JSON string into a Python dictionary
    document = clean_document(description)

    print(embedding_model)

    gen_embeddings = None

    if embedding_model == "text_embedding-3-small":
        #collection = db['openai_documents']
        gen_embeddings = openai_embedding(json.dumps({
            'name' : document['name'],
            'summary' : document['summary']
        }))
    elif embedding_model == "amazon.titan-embed-text-v2:0":
        #collection = db['gemini_documents']
        gen_embeddings =  get_embedding_from_titan_multimodal(json.dumps({
            'name' : document['name'],
            'summary' : document['summary']
        }))

    elif embedding_model == "models/embedding-001":
       # collection = db['gdocuments']
        gen_embeddings = get_embedding_from_gemini(json.dumps({
            'name' : document['name'],
            'summary' : document['summary']
        }))
        
            

   
    collection.insert_one({
        'image': encoded_image,
        'api_key': st.session_state.api_code,
        'embedding' : gen_embeddings,
        'ocr': document,
        'ai_tasks': []
    })

def open_ai_gen_task(ocr_text,prompt):
    ## Use existing document as context and perform another GPT task
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{
        "role": "system",
        "content": "You are  a task assistant looking to create a task for the AI to perform on the JSON object. Please return plain output which is only copy paste with no explanation."
        },
        {
        "role": "user",
        "content": f"Please perform the following task {prompt}  on the following JSON object {ocr_text}. Make sure that the output is stright forward to copy paste."
        }
        ]
        )
    
    return response.choices[0].message.content

def claude_gen_task(ocr_text,prompt):
    ## Use existing document as context and perform another GPT task
    response = bedrock_runtime.invoke_model(
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1000,
            "system": f"You are  a task assistant looking to create a task for the AI to perform on the JSON object. Please return plain output which is only copy paste with no explanation.",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Please perform the following task {prompt}  on the following JSON object {ocr_text}. Make sure that the output is stright forward to copy paste."}
                ]
            }]
        }),
        modelId="anthropic.claude-3-5-sonnet-20240620-v1:0",
        accept="application/json",
        contentType="application/json",
    )

    response_body = json.loads(response.get("body").read())
    return response_body["content"][0].get("text", "No description available")

def gemini_gen_task(ocr_text,prompt):

    gemini = genai.GenerativeModel('gemini-pro')
    response = gemini.generate_content([f"You are  a task assistant looking to create a task for the AI to perform on the JSON object. Please return plain output which is only copy paste with no explanation.", f"Please perform the following task {prompt}  on the following JSON object {ocr_text}. Make sure that the output is stright forward to copy paste."], stream=False)
    return response.text

def get_ai_task(ocr,prompt,model):
    ## Use existing document as context and perform another GPT task
    ocr_text = json.dumps(ocr)
    if (model == "gpt-4o"):
        response = open_ai_gen_task(ocr_text,prompt)
        return response
    elif(model == "gemini-pro"):
        response = gemini_gen_task(ocr_text,prompt)
        return response
    elif(model == "bedrock-claude-3"):
        response = claude_gen_task(ocr_text,prompt)
        return response

def save_ai_task(task_id, task_result, prompt):

    collection.update_one(
        {"_id": ObjectId(task_id)},
        {"$push" : {"ai_tasks" : {'prompt' : prompt, 'result' : task_result}}}
    )

    return "Task saved successfully."


def openai_chat(query,message):
    relevant_docs = vector_search_aggregation(query, 3, 'text-embedding-3-small')
    context = ''
    for doc in relevant_docs:
        context+=json.dumps(doc['ocr'])
    messages=[{"role": "system", "content": "You are an assistant that uses document context to answer questions. Answer not too long and concise answers."}]
    for chat_message in st.session_state.messages:
        messages.append(chat_message)

    messages.append({"role": "user", "content": f"Using the following context, please answer the question: {query}\n\nContext:\n{context}"})
    
    stream = openai.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        stream=True
    )
    response = message.write_stream(stream)

    st.session_state.messages.append({"role": "assistant", "content": response})

def gemini_chat(query,message):

    relevant_docs = vector_search_aggregation(query, 3, 'models/embedding-001')
    context = ''
    for doc in relevant_docs:
        context += " " + str(doc['ocr']) + "\n\n"
    
    
    response = gemini_pro.generate_content([f"Using the following context : {context}\n\n, Question: {query}\n\n Provide more details to user"], stream=False)

    #response=chat.send_message(f"Using the following context : {context}\n\n, Question: {query}\n\n Provide more details to user",stream=False)
    message.markdown(response.text)
    # message.markdown(response.text)

    st.session_state.messages.append({"role": "assistant", "content": response.text})

def claude_chat(query,message):

    relevant_docs = vector_search_aggregation(query, 3, 'amazon.titan-embed-text-v2:0')
    context = ''
    for doc in relevant_docs:
        context += " " + str(doc['ocr']) + "\n\n"
    
    response = generate_claude_response(context, query)
    message.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

def generate_claude_response(context, query):

    claude_body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1000,
        "system": "You are an assistant that uses document context to answer questions. Answer not too long and concise answers.",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": f"Using the following context : {context}\n\n, Question: {query}\n\n Provide more details to user"}
            ]
        }]
    })

    claude_response = bedrock_runtime.invoke_model(
        body=claude_body,
        modelId="anthropic.claude-3-sonnet-20240229-v1:0",
        accept="application/json",
        contentType="application/json",
    )
    response_body = json.loads(claude_response.get("body").read())
    return response_body["content"][0].get("text", "No description available")
    
    

def search_aggregation(search_query):
    docs = list(collection.aggregate([
        {
            '$search': {
                'index': 'search', 
                'compound': {
                    'should': [
                        {
                            'text': {
                                'query': search_query, 
                                'path': {
                                    'wildcard': '*'
                                }
                            }
                        }
                    ], 
                    'filter': [
                        {
                            'queryString': {
                                'defaultPath': 'api_key', 
                                'query': st.session_state.api_code
                            }
                        }
                    ]
                }
            }
        }
    ]))
    return docs   

def vector_search_aggregation(search_query, limit, model):
    print(model)
   
    query_vec = None
    if model == "text-embedding-3-small":
        query_vec = openai_embedding(search_query)
    elif model == "amazon.titan-embed-text-v2:0":
        query_vec = get_embedding_from_titan_multimodal(search_query)
    elif model == "models/embedding-001":
        query_vec = get_embedding_from_gemini(search_query)
    
    if query_vec is None:
        return []

    docs = list(collection.aggregate([
        {
            '$vectorSearch': {
                'index': 'vector_index', 
                'queryVector': query_vec, 
                'path': 'embedding', 
                'numCandidates' : 20,
                'limit' : limit,
                'filter': {
                    'api_key': st.session_state.api_code
                }
            }},
            { '$project' : {'embedding'  : 0} }
    ]))
    return docs


# Main app logic
if not st.session_state.authenticated:
    auth_form()
else:
    st.title("👀 AllCR App")

    

    # Image capture
    col1, col2 = st.columns(2)
    with col1:
        st.header("Capture Objects with AI")
    with col2:
        model = st.selectbox("Choose the model", [("gpt-4o", "text-embedding-3-small"), ("gemini-pro", "models/embedding-001"), ("bedrock-claude-3", "amazon.titan-embed-text-v2:0")])
        if model[0] == "gpt-4o":
            st.write("You have selected the GPT-4o model.")
            analyser="GPT"
            collection = db['openai_documents']
            st.session_state.messages=[]
        elif model[0] == "gemini-pro":
            st.write("You have selected the Gemini Pro model.")
            analyser="Gemini Vision"
            collection = db['gemini_documents']
            st.session_state.messages=[]
        elif model[0] == "bedrock-claude-3":
            st.write("You have selected the Bedrock Claude 3 model.")
            analyser="Claude3 Vision"
            collection = db['bedrock_documents']
            st.session_state.messages=[]
        else:
            st.write("You have selected the default model - GPT-4o.")

    st.divider()
    st.write("Capture real life objects like Recipes, Documents, Animals, Vehicles, etc., and turn them into searchable documents.")
    
   
    options = st.multiselect(
        "What do you want to capture?",
        ["Recipe", "Post", "Screenshot","Document", "Animal", "Vehicle", "Product", "Sports", "Other"], ["Other"])

    transcribed_object = options[0] if options else "other"
    tab_cam, tab_upl = st.tabs(["Camera", "Upload"])
    with tab_cam:
        image = st.camera_input("Take a picture")

    with tab_upl:
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
        # To read file as bytes:
            image = uploaded_file

    @st.experimental_dialog("Processed Document",width="large")
    def show_dialog():
        st.write(extracted_text)
        if st.button("Confirm Save to MongoDB"):
        
            save_image_to_mongodb(img, extracted_text, model[1])
            st.rerun()
            
    @st.experimental_dialog("AI Task on Document",width="large")
    def show_prompt_dialog(work_doc):
        st.header("Please describe the AI processing to be done on the document.")
        st.markdown(f"""### Document: {work_doc['ocr']['name']} 
                                 
                                 Example: Translate this document to French.
                                 """)
        prompt = st.text_area("AI Prompt")
        if st.button("Confirm task"):
            result = get_ai_task(work_doc['ocr'],prompt,model[0])
            st.code(result)
            res = save_ai_task(work_doc['_id'], result, prompt)
            st.success(res)
            work_doc['ai_tasks'].append({'prompt' : prompt,
                                        'result' : result})
            # if st.button("Save Task to Document"):
        ## if length of array bigger than 0
        if 'ai_tasks' in work_doc and len(work_doc['ai_tasks']) > 0:
            st.markdown("### Previous Tasks")
            for task in work_doc['ai_tasks']:
                with st.expander(f"Task: {task['prompt']}"):
                    text, markdown = st.tabs(["text", "markdown"])
                    with text:
                        st.markdown(task['result'])
                    with markdown:
                        st.code(task['result'])
        else:
            st.write("No previous tasks found.")
                
                
                 


    if st.button("Analyze image for MongoDB"):
        if image is not None:
            with st.spinner(f"Analysing document with {analyser}..."):
                img = Image.open(io.BytesIO(image.getvalue()))
                extracted_text = transform_image_to_text(img, img.format, model[0])
            show_dialog()
            

    # Search functionality
    with st.sidebar:
        st.header("Chat with AI")

        if st.button("New Chat"):
            st.session_state.messages=[]
       
        messages = st.container(height=500)
        for message in st.session_state.messages:
            
            with messages.chat_message(message["role"]):
                messages.markdown(message["content"])

        # Accept user input
        if prompt := st.chat_input("Ask me something about the docs..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            # Display user message in chat message container
            with messages.chat_message("user"):
                messages.markdown(prompt)
            with st.spinner('RAGing...'):
                
                with messages.chat_message("assistant"):
                    if model[0] == "gpt-4o":
                        openai_chat(prompt,messages)
                    elif model[0] == "gemini-pro":
                        gemini_chat(prompt,messages)
                    elif model[0] == "bedrock-claude-3":
                        claude_chat(prompt,messages)
    

    ## Adding search bar
    search_query = st.text_input("Search for documents")
    toggle_vector_search = st.toggle("Vector Search", False)
    if search_query:
        if not toggle_vector_search:
            docs = search_aggregation(search_query)
        else:
            docs = vector_search_aggregation(search_query, 5, model[1])
    else:
        docs = list(collection.find({"api_key": st.session_state.api_code}).sort({"_id": -1}))
    for doc in docs:
        expander = st.expander(f"{doc['ocr']['type']} '{doc['ocr']['name']}'")
        expander.write(doc['ocr'])  # Ensure 'recipe' matches your MongoDB field name
        ## collapseble image

        image_col, prompt_col = expander.columns(2)
        
        with image_col:
            if expander.button("Show Image", key=f"{doc['_id']}-image"):
                image_data = base64.b64decode(doc['image'])
                image = Image.open(io.BytesIO(image_data))
                expander.image(image, use_column_width=True)

        with prompt_col:
            if expander.button("Run AI Prompt", key=f"{doc['_id']}-prompt"):
               show_prompt_dialog(doc)

   
