# 👀 AllCR App

OCR App is a Streamlit application that allows users to capture real-life objects like recipes, documents, animals, vehicles, and more, and turn them into searchable documents. The app integrates with OpenAI's GPT-4, AWS Bedrock and Google Gemini for OCR (Optical Character Recognition) to JSON conversion and MongoDB Atlas for storing the extracted information.

This demo showcase the power of MongoDB and it being LLM agnostic. 

## Supported AI sets:

- **LLM**:`OpenAI gpt-4o` **Embedding Model**: `text-embedding-3-small`
- **LLM**:`Google gemini-pro` **Embedding Model**: `models/embedding-001`
- **LLM**:`Bedrock Claude3`  **Embedding Model**: `amazon.titan-embed-text-v2:0`

## Features

- **Authentication**: Secure access to the application using an API code.
- **Image Capture**: Capture images using your device's camera.
- **OCR to JSON**: Convert captured images to JSON format using OpenAI's GPT-4.
- **MongoDB Integration**: Store and retrieve the extracted information from MongoDB.
- **Search and Display**: Search and display stored documents along with their images.
- **Chat with AI**: Open the sidebar to chat with GPT on the context captured by the app.

## Requirements

- Python 3.8+
- Streamlit
- OpenAI Python Client Library
- MongoDB Atlas cluster

Once the cluster is deployed perform the following tasks:
1. Create a database named 'ocr_db' with collection 'api_keys' :
```
use ocr_db
db.api_keys.insertOne({'api_key' : "<YOUR_IMAGINARY_KEY>"});
```
2. Create a 2 search indexes:
2.1 [Vector search](https://www.mongodb.com/docs/atlas/atlas-vector-search/tutorials/vector-search-quick-start/) index on -  

OPEN AI : 'ocr_db.openai_documents':

```
{
  "fields": [
    {
      "numDimensions": 1536,
      "path": "embedding",
      "similarity": "cosine",
      "type": "vector"
    },
    {
      "path": "api_key",
      "type": "filter"
    }
  ]
}
```

Google Gemini: 'ocr_db.gemini_documents':

```
{
  "fields": [
    {
      "numDimensions": 768,
      "path": "embedding",
      "similarity": "cosine",
      "type": "vector"
    },
    {
      "path": "api_key",
      "type": "filter"
    }
  ]
}
```

AWS Bedrock : 
```
{
  "fields": [
    {
      "numDimensions": 1024,
      "path": "embedding",
      "similarity": "cosine",
      "type": "vector"
    },
    {
      "path": "api_key",
      "type": "filter"
    }
  ]
}
```

2.2 Atlas [text Search](https://www.mongodb.com/docs/atlas/atlas-search/tutorial/create-index/) index on 'openai_documents', 'bedrock_documents', 'gemini_documents':
```
{
  "mappings": {
    "dynamic": true,
    "fields": {
      "api_key": {
        "type": "string"
      },
      "ocr": {
        "dynamic": true,
        "type": "document"
      }
    }
  }
}
```


- PIL (Python Imaging Library)



## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/allcr-app.git
   cd allcr-app
   ```

   ## Installation



2. **Install the required packages:**
```
   pip install -r requirements.txt
```
3. **Set up environment variables:**

   Create a `.env` file in the root directory of your project and add your OpenAI API key, MongoDB URI, and API code for authentication.
```
   ## Open AI
   OPENAI_API_KEY=your_openai_api_key

   ## Bedrock Credentials
   AWS_ACCESS_KEY=your_aws_access_key
   AWS_SECRET_KEY=your_aws_secret_key

   ## Google Credentials
   GOOGLE_API_KEY=your_google_api_key

   ## MongoDB URI
   MONGODB_ATLAS_URI=your_mongodb_atlas_uri
```
## Usage

1. **Run the Streamlit app:**
```
   streamlit run app.py
```
2. **Access the app:**
```
   Open your web browser and go to `http://localhost:8501`.
```
Once prompted input the api_key saved in Atlas under the 'ocr_db.api_keys' collection.

3. **Authenticate:**

   Enter the API code provided in your `.env` file to access the application.

4. **Capture and Process Images:**
   - Choose the relevant AI modules to use in the dropdown.
   - Select the type of object you want to capture.
   - Use the camera to take a picture of the object.
   - The image will be processed, and the extracted text will be displayed for confirmation.
   - Save the processed document to MongoDB.

5. **Search and Display Documents:**

   - Use the search functionality to find stored documents.
   - Expand the results to view the extracted text and display the associated image.

## Code Overview

- **`app.py`**: Main application script that contains the Streamlit app logic.
- **`requirements.txt`**: List of required Python packages.

## Key Functions

- **`auth_form()`**: Handles user authentication using an API code.
- **`transform_image_to_text(image)`**: Transforms a captured image to text using OpenAI's GPT-4.
- **`save_image_to_mongodb(image, description)`**: Saves the captured image and extracted text to MongoDB.
- **`search_and_display_images(query)`**: Searches and displays images from MongoDB based on the query.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For questions or suggestions, please contact [Pavel](mailto:pavel.duchovny@mongodb.com).
