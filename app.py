from flask import Flask, render_template, request, redirect, url_for, session
import pickle
import pandas as pd
from surprise import SVD
from dotenv import load_dotenv
import os

# Setup Flask
app = Flask(__name__)
load_dotenv()

# api key
openai.api_key = os.getenv("super_secure_random_key")

# Load model and data
model = pickle.load(open("svd_model.pkl", "rb"))
books = pd.read_csv("books.csv")
ratings = pd.read_csv("ratings.csv")

# Genre assignment function
def assign_genre(title):
    if not isinstance(title, str):
        return 'general'
    title = title.lower()
    genres = {
        'fantasy': ['fantasy', 'magic', 'wizard', 'dragon'],
        'thriller': ['thriller', 'suspense', 'mystery'],
        'psychology': ['psychology', 'mind', 'behavior'],
        'science fiction': ['science fiction', 'sci-fi', 'space']
    }
    for genre, keywords in genres.items():
        if any(keyword in title for keyword in keywords):
            return genre
    return 'general'

# Add genre to books
books['Genre'] = books['Book-Title'].apply(assign_genre)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            input_title = request.form["book_title"].strip().lower()
            input_rating = float(request.form["rating"])
            input_genre = request.form["genre"].strip().lower()

            if not 0 <= input_rating <= 10:
                raise ValueError("Rating must be between 0 and 10")

            session.clear()

            # Match input title
            matched_books = books[books["Book-Title"].str.lower().str.contains(input_title)]
            if matched_books.empty:
                return render_template("index.html", error="Book not found. Try another title.")

            book_isbn = matched_books.iloc[0]["ISBN"]
            book_title = matched_books.iloc[0]["Book-Title"]

            # Add user's rating to a temporary copy of ratings
            temp_ratings = ratings.copy()
            temp_ratings.loc[len(temp_ratings)] = [99999, book_isbn, input_rating]

            # Filter genre-matching books
            genre_books = books[(books["Genre"] == input_genre) & (books["ISBN"] != book_isbn)]

            # Get average ratings
            avg_ratings = ratings.groupby("ISBN")["Book-Rating"].mean().reset_index()
            avg_ratings.columns = ["ISBN", "Avg-Rating"]

            # Merge average ratings with genre books
            genre_books = pd.merge(genre_books, avg_ratings, on="ISBN", how="left")

            # Create recommendation list with adjusted scores
            recommendations = []
            for _, book in genre_books.iterrows():
                try:
                    if pd.notnull(book["Avg-Rating"]):
                        pred = model.predict(99999, book["ISBN"])
                        adjusted_score = (pred.est + book["Avg-Rating"]) / 2
                        recommendations.append({
                            'title': book["Book-Title"],
                            'isbn': book["ISBN"],
                            'score': adjusted_score,
                            'avg_rating': book["Avg-Rating"]
                        })
                except Exception as e:
                    print("Prediction error:", e)
                    continue

            if recommendations:
                recommendations.sort(key=lambda x: x['score'], reverse=True)
                final_recs = recommendations[:5]

                session['recommendations'] = [
                    (r['title'], round(r['score'], 2)) for r in final_recs
                ]
                session['input_rating'] = input_rating
                return redirect(url_for('result'))

            # Fallback to average ratings
            books_with_ratings = pd.merge(genre_books, avg_ratings, on="ISBN", suffixes=('', '_dup'))
            rating_range = (input_rating - 1.5, input_rating + 1.5)
            filtered = books_with_ratings[
                books_with_ratings["Avg-Rating"].between(*rating_range)
            ].sort_values("Avg-Rating", ascending=False)

            if len(filtered) < 5:
                filtered = books_with_ratings.sort_values("Avg-Rating", ascending=False)

            session['recommendations'] = [
                (row["Book-Title"], round(row["Avg-Rating"], 2))
                for _, row in filtered.head(5).iterrows()
            ]
            session['input_rating'] = input_rating
            return redirect(url_for('result'))

        except ValueError as e:
            return render_template("index.html", error=str(e))
        except Exception as e:
            print("Unexpected error:", e)
            return render_template("index.html", error="An error occurred. Please try again.")

    return render_template("index.html")

@app.route("/results")
def result():
    recommendations = session.get('recommendations', [])
    input_rating = session.get('input_rating', None)

    if not recommendations:
        return render_template("results.html",
                               message="No recommendations found.",
                               input_rating=input_rating)

    return render_template("results.html",
                           recommendations=recommendations,
                           input_rating=input_rating)

if __name__ == "__main__":
    app.run(debug=True)
