
# Original Project Title and Summary 

Audiohead Vibes, the initail music recommender systems goal was to recommend music to a user. The way it worked was it recommended songs based on a user's preferences. The songs were ranked by score based on how well it matched the user's prefences. The top five songs with the highest scores were listed. 

# Title (updated music recommender system)

Vector Vibes

This system uses Retrival-Augmented Generation (RAG) to recommend music from a user's text input. The system takes that query into audio features (energy, danceability, mood, valence, acoustocness, tempo, loudness and genre). The system then checks that these values are in the dataset and transforms them into a target vector.

Then the system compares the target vector to the songs across 10 geners using consine similarity. The 10 closest matches are returned back to Claude. Claude then picks returns the best song with a reason based on the song data. 

The result is returned as a song card in a Streamlit app. A user can like or dislike a song, and skip a song. Liking a song will create an update to a user's taste profile.  

# Architecture Overview

The diagram is a top to bottom flowchart that shows the system's three main stages: input, the AI process, and output. 

The input is the user's text request. The AI process is when Claude turns the text requet into audio features that become the target vector. The vector blends with the user's taste profilet to find 10 of the closets songs and return the top match. The output is the song card. 

# Setup Instructions 

Requirements: Python 3.10 or newer and an Anthropic API key. 

1. Open the project folder in a terminal 
```
cd rag-music-recommender
```

2. Create and activate a virtual enviroment 

```
python -m venv .venv 
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows 
```
3. Install dependancies 

```
pip install -r requirements.txt 
```

4. Add your API key 
    Copy the example file, then open `.env` and replace `your-api-keu-here` with your actual key: 

    ```
    cp .env.example .env
    ```

    `.env` is in `.gitignore`, so your key will not be committed. 

5. Run the app
   ```
   streamlit run app.py 
   ```
    Streamlit opens the app in your browser. Type a request like, "chill pop songs for studying" and click Recommend. 

# Sample Interactions 

Input: "chill study music"

| # | Song | Artist(s) | Genre | Energy | Valence | Tempo | Score |
|---:|---|---|---|---:|---:|---:|---:|
| 1 | Little Drummer Boy | Scott Hamilton | jazz | 0.38 | 0.34 | 92 | 0.992 |
| 2 | Love Letter - Original Mix | Living Room | jazz | 0.30 | 0.42 | 100 | 0.988 |
| 3 | Don't Cry Baby | Living Room | jazz | 0.41 | 0.61 | 75 | 0.981 |
| 4 | Feelin' Good | Zmeyev | jazz | 0.29 | 0.58 | 80 | 0.980 |
| 5 | Evening Silence | Three Four Trio | jazz | 0.14 | 0.11 | 89 | 0.978 |
| 6 | Autumn In New York | Bobby Timmons | jazz | 0.11 | 0.15 | 69 | 0.976 |
| 7 | Linus And Lucy | Vince Guaraldi Trio | jazz | 0.39 | 0.69 | 82 | 0.974 |
| 8 | Christmas Time Is Here - Instrumental | Vince Guaraldi Trio | jazz | 0.12 | 0.09 | 75 | 0.973 |
| 9 | The Beauty of Everyday Things | Hara Noda | jazz | 0.28 | 0.09 | 108 | 0.971 |
| 10 | Nuit Douce | Phlocalyst, Myríad | jazz | 0.22 | 0.16 | 128 | 0.968 |


Input: "hype workout music"

| # | Song | Artist(s) | Genre | Energy | Valence | Tempo | Score |
|---:|---|---|---|---:|---:|---:|---:|
| 1 | Ud-daa Punjab | Vishal Dadlani, Amit Trivedi | hip-hop | 0.91 | 0.62 | 122 | 0.997 |
| 2 | They Know | Karan Aujla | hip-hop | 0.72 | 0.61 | 140 | 0.996 |
| 3 | Celebrity Killer (feat. Tion Wayne) | Sidhu Moose Wala, Tion Wayne | hip-hop | 0.72 | 0.59 | 144 | 0.993 |
| 4 | Sirikkalam Parakkalam | Benny Dayal, Madurai Soulijos | hip-hop | 0.83 | 0.54 | 128 | 0.993 |
| 5 | Calaboose | Sidhu Moose Wala | hip-hop | 0.76 | 0.56 | 150 | 0.993 |
| 6 | Two Two Two (From "Kaathuvaakula Rendu Kaadhal") | Anirudh Ravichander, Sunidhi Chauhan, Sanjana Kalmanje | hip-hop | 0.69 | 0.65 | 140 | 0.993 |
| 7 | Jalabulajangu (From "Don") | Anirudh Ravichander, Rokesh | hip-hop | 0.82 | 0.46 | 151 | 0.993 |
| 8 | One Side | DIVINE | hip-hop | 0.81 | 0.55 | 92 | 0.992 |
| 9 | Betty (Get Money) | Yung Gravy | hip-hop | 0.77 | 0.58 | 102 | 0.992 |
| 10 | Sigan Bailando | Wisin & Yandel, Tego Calderón, Franco "El Gorilla" | hip-hop | 0.83 | 0.68 | 180 | 0.992 |

Input: "sad breakup songs" 

| # | Song | Artist(s) | Genre | Energy | Valence | Tempo | Score |
|---:|---|---|---|---:|---:|---:|---:|
| 1 | Alguém Que Te Faz Sorrir | Fresno | r-n-b | 0.51 | 0.24 | 104 | 0.959 |
| 2 | Thodi Jagah (From "Marjaavaan") | Arijit Singh, Tanishk Bagchi | pop | 0.47 | 0.23 | 107 | 0.959 |
| 3 | Yağmurlar | Hüsnü Arkan, Feyza Eren | jazz | 0.39 | 0.23 | 96 | 0.959 |
| 4 | I Won't Let Go | Rascal Flatts | country | 0.41 | 0.22 | 132 | 0.957 |
| 5 | Jitni Dafa | Yasser Desai, Jeet Gannguli, Rashmi Virag | pop | 0.51 | 0.25 | 96 | 0.957 |
| 6 | I Feel Like I'm Drowning | Two Feet | rock | 0.58 | 0.23 | 105 | 0.957 |
| 7 | Humdard | Arijit Singh | pop | 0.54 | 0.27 | 87 | 0.956 |
| 8 | Asal Mein | Darshan Raval | pop | 0.55 | 0.33 | 114 | 0.955 |
| 9 | Little Drummer Boy | Mary J. Blige | soul | 0.42 | 0.20 | 96 | 0.955 |
| 10 | Read All About It, Pt. III | Emeli Sandé | soul | 0.44 | 0.34 | 98 | 0.955 |


# Design Decisions 

1. Content-base filtering and not collaborative 

The choice to use content-filtering was because there was no user listening data. The tradeoff for this decision was that the recommender only finds songs that sound similar. 

2. Haiku as the model 


3. Artist cap and duplication 

# Testing Summary 

```
OVERALL EVALUATION
------------------------------
Songs evaluated: 3856
Average same-genre rate: 0.998
Average cosine similarity: 0.993
Average unique genres in top 5: 1.01

RESULTS BY GENRE
------------------------------
             same_genre_rate  average_similarity  unique_genres
track_genre                                                    
classical              0.998               0.991          1.007
country                0.997               0.994          1.003
electronic             1.000               0.992          1.002
hip-hop                0.995               0.994          1.010
jazz                   1.000               0.989          1.000
pop                    0.997               0.993          1.013
r-n-b                  1.000               0.994          1.000
reggae                 1.000               0.995          1.000
rock                   0.995               0.993          1.012
soul                   0.993               0.991          1.015
```