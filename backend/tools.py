import json 
import logging 
from typing import Any , Dict, List 

from pydantic import BaseModel ,Field , validate_call
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logging.basicConfig(level=logging.INFO)
logger =logging.getLogger(__name__)

# ---------- Load model once (global) ----------
analyzer = SentimentIntensityAnalyzer()

# ---------- Input/Output schemas ----------

class SentimentInput(BaseModel):
    text: str = Field(... ,description="the text to analyse sentiment of .")


class SentimentOutput(BaseModel):
    sentiment : str                #positive , negative , neural
    confidence : float             #between 0 and 1 
    scores : Dict[str , float]     #raw scores for transparency

    #----------------tool function ----------------

@validate_call
def sentiment_analysis(data: SentimentInput) -> str :
    """ Analyse the sentiment of the given text.
    Returns sentiment lable (positive/negative/neutral) with confidencce score.
    Use this tool whenever the user asks about sentiment , emotion ,or openion in the text 
    """
    try:
        scores= analyzer.polarity_scores(data.text)
        compound =scores['compound']

    #decide label

    if compound >=0.05:
        sentiment = "positive"
    elif compound <= -0.05:
        sentiment = "negative"
    else :
        sentiment ="neutral"
        