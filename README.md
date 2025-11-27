# Rhoda QnA Bot

AI-powered chatbot for robot annotation guidelines and FAQ. Helps the  team get instant answers to their questions.

## Features

- Ask questions about robot annotation guidelines
- Get instant AI-powered answers
- Beautiful web interface
- Powered by OpenAI GPT-4 and Pinecone vector database

## Setup

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Create `.env` file with your API keys:
```
   OPENAI_API_KEY=your_key_here
   PINECONE_API_KEY=your_key_here
   PINECONE_INDEX_NAME=robot-annotation-bot-index
   PORT=8000
```
4. Upload documents to Pinecone: `python ingest_documents.py`
5. Run the bot: `python main.py`
6. Open `index.html` in your browser

## Documentation

See the annotation guidelines in the `documents/` folder.

## Security

Never commit the `.env` file to GitHub. API keys must remain private.
```

3. **Save** (Ctrl + S)

---

### **Step 4: Create `.env.example`**

This shows what the `.env` file should look like (without real keys).

1. **Create new file**: `.env.example`
2. Paste this:
```
OPENAI_API_KEY=your_openai_key_here
PINECONE_API_KEY=your_pinecone_key_here
PINECONE_INDEX_NAME=robot-annotation-bot-index
PORT=8000