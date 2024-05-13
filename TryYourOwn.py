from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.pipeline import make_pipeline

import pandas as pd

# load back the pipeline
pipeline = joblib.load('text_processing_pipeline.joblib')

# TYPE IN HERE YOUR OWN REVIEW FOR PREDICTION
X_test = "As a John Woo fan I really thought this was going to be his comeback. The concept was so cool, and the trailer looked so good. I genuinely felt like this had so much potential and I was so excited to see it realized. I kept imagine how intense sequences of pure action must be without any interruption of dialogue. I kept imagining the unique focus on the visuals and sounds to replace the work of the dialogue. Mr. Robot pulled off a thrilling episode with no dialogue, so surely this movie could've too. Truly there was so much that could've been done and this would've been a classic. Unfortunately that trailer literally showed the best moments, and not much was done with the concept. All they really had to do was follow the John Wick blueprint but focus more on the visual storytelling and give it a unique angle with the mute protagonist on top of the lack of dialogue that could've made this movie so much more tense. Unfortunately, the lack of dialogue detracts from the characterization and feels forced in many scenes. I feel like there could've been a lot more done with the protagonist who's very well performed, but not well written. I question his willingness to go on shooting sprees when his son was killed by a stray bullet. There is nothing really done with the protagonist suddenly becoming mute, and the drama with his wife has no real development. The mandatory set up for the emotional beats to hit are executed in such a boring and cliche way and takes up what feels like half the film. Every scene with the son felt so repetitive, and like it was begging you to care with the music, but I really didn't. It almost became cheesy, especially at the end. There's this cop character that's just... there, I guess. There's no personality given to the villains except it keeps cutting to the main villain drugging and shagging this random girl... When we finally get to the action John Woo does deliver, but it's not his best work, and doesn't redeem such a thin story. There's moments of Woo's brilliance shining through what are mostly just pretty good action sequences by today's standards. A lot of the car scenes weren't the best either. The action has its moments and the visuals are often cool, but even then there are some odd directorial choices like overuse of slow-mo and some odd editing choices. I always just felt like there's this great action scene coming, and some parts were, but other parts were off. What's worse is that much of the action feels completely unbelievable and just dumb with the protagonist, of course, getting along through luck and pure incompetence from the villains. The last 2 confrontations in this movie are what really tanked it for me to call it bad. It's just plain ridiculous and unsatisfying how they unfold. Genuinely feels like some AI choreography, unreal, and it's so stupid character wise. With all this being said I still mostly enjoyed the movie thanks to some good action and visuals, but I felt like it was more due to the goodwill I have for John Woo, like I owe him for his older awesome films. This was overall a disappointing film and such a waste of the concept. I hope someone else tries something like this again and does better. I don't want to be too hard on Mr. Woo though. Please John Woo, you are better than this, and we still love you!"
#------------------
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

# Load stopwords once
stop_words = set(stopwords.words('english'))

# Function to clean and preprocess text
def preprocess_text(text):
    # Lowercase, remove HTML tags, non-alphabets, and numbers
    text = re.sub(r'<.*?>', '', text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Tokenization and stopword removal
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]

    return ' '.join(tokens)


X_test = preprocess_text(X_test)

#------------------

# Predict the class of a new review
y_pred = pipeline.predict([X_test])

# Print the prediction
print(y_pred)
