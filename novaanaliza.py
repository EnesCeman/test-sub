import os
import matplotlib.pyplot as plt
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib.ticker import PercentFormatter

load_dotenv(find_dotenv())
client = OpenAI()

def truncate_text(text, max_length):
    if len(text) > max_length:
        return text[:max_length] + "..."
    return text


def generate_recommendation(business_data):
    
    # Generate prompt for ChatGPT based on reviews and sentiment scores
    prompt = ""
    for index, row in business_data.iterrows():
        review = row["Review"]
        sentiment_score = row["Sentiment Score"]
        # Truncate the review if it exceeds a certain length
        review = truncate_text(review, 1200)
        prompt += f"Review: {review}\nSentiment Score: {sentiment_score}\n"

    # Truncate prompt to a maximum length of 4096 tokens
    prompt = truncate_text(prompt, 4096)

    # Generate completion using OpenAI API
    # client.api_key = 
    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt
        + "\nWrite me similar mail, that would get the same or higher sentiment score",
        max_tokens=974,  # Adjusted max_tokens for a longer response
    )

    if response.choices[0].finish_reason == "stop":
        recommendation = response.choices[0].text.strip()
    else:
        print("Response details:", response)
        recommendation = "Error generating recommendation"

    return recommendation


def plot_histogram(business_data):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(
        business_data["Sentiment Score"],
        edgecolor="black",
        weights=np.ones_like(business_data["Sentiment Score"])
        * 100
        / len(business_data),
    )
    plt.xlabel("Sentiment Score")
    plt.ylabel("Relative Frequency (%)")
    ax.yaxis.set_major_formatter(PercentFormatter())
    st.pyplot(fig)


def main():
    st.title("Mail Recommendation")

    # Upload CSV file
    file = st.file_uploader("Upload a CSV file", type=["csv"])

    if file is not None:
        if file is not None:
            if not file.name.endswith(".csv"):
                st.error("Error: Please upload a CSV file.")
                return
            try:
                business_data = pd.read_csv(file)

                # Display uploaded data
                st.subheader("Uploaded Mail Data")
                st.write(business_data)

                # Check if required columns exist
                required_columns = ["Review", "Sentiment Score"]
                if set(required_columns).issubset(business_data.columns):
                    st.subheader("Mail Recommendation")
                    recommendation = generate_recommendation(business_data)

                    # Display recommendation
                    st.write("Summary Recommendation:")
                    st.write(str(recommendation))  # Convert recommendation to string
                    # Display Mail
                    st.subheader("Mail Content:")
                    mail_content = "\n".join(
                        f"{index + 1}. {row['Review']}"
                        for index, row in business_data.iterrows()
                    )
                    st.text(mail_content)
                    # Plot relative frequency histogram
                    st.subheader("Sentiment Score Distribution")
                    plot_histogram(business_data)
                else:
                    st.write(
                        "Error: The uploaded CSV file does not contain all the required columns."
                    )
            except pd.errors.EmptyDataError:
                st.error("Error: The uploaded CSV file is empty.")
            except Exception as e:
                st.error(f"An error occured: {e}")

if __name__ == "__main__":
    main()