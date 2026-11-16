# 🎧 Model Card: RAG Music Recommender Simulation

## 1. Model Name 

Vector Vibes 

## 2. Intended Use 

Vector vibes is application that recommends songs from a internal dataset based on user input like, "chill songs for studying." 

The system finds the 10 closest songs using cosine similarity and user's taste. Claude picks the best matches based on the song data.

## 3. How does the Model Works 

Vector Vibes runs each request in four steps: 
    1. Parse using Claude Haiku
    2. Builds the target vector using audi features  
    3. Using cosine similarity to compare target agians all 5,000 songs and keeps the 10 closest. 
    4. From the 10 songs Claude chooses the best fits (5 shown by default) and writes one sentance for each. 

## 4. Data


## 5. Strengths  

Recommendations cannot be hallucinated. Claude can only choose from the 10 songs that the retrevial found. The guardrails remove anything that Claude attemps to create. 

The system can handle plain language. 

Retrieval matches strongly align with genre. 

## 6. Limitations and Bias

The recommender rarely crosses genres. The evaluation found that 99.8% of same genre recommmndations. The catalog is fixed. The taste profile are not persistant. They only last during the streamlit session.   

## 9. Personal Reflection

When I initially began his projet I collaborated with ChatGPT to choose a RAG approach. Initially ChatGPT suggested that I use collaborative filtering which I thought was overkill and beyond the scope of this project. This is an example of AI over-engineering. I chose to go with content-based filtering because there aren't multiple users. 


I asked the AI to transform the dataset to drop duplicated songs. Despite filtering there the duplicates persisted. So, I switched to Claude Code. This also prompted me to include tests for duplicate songs in my dataset. 

Content based filtering because their is not user data beyond a user's preferences. Collaborative filtering learns from a user's history. Content based filtering only needs song features. 

Claude Code recommendation that I included in my project was using gaurdrails to prevent Caude from hallucinating recommendations. This helped Claude's response stay ancored to the dataset. 

A recommendation that I decided against was commiting my API key. I thought this was a strange recommendation because Claude warns users about entering API keys in the chat, but is seemingly not concerned with commiting an API key to a public repo. This is an obvious security concern.

I told Claude that I would store my API key in a seperate `.env` file and include a reference to the `.env` file in my `.gitignore`. 
