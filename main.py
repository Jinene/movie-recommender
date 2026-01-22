from recommend import recommend, load_data

# Load movie data
movies, ratings = load_data()



print("🎬 Welcome to the Movie Recommendation System! 🎬")
movie_name = input("Enter a movie name: ")

try:
    recommendations = recommend(movie_name, movies)
    print("\nTop 10 movie recommendations:")
    for i, movie in enumerate(recommendations, 1):
        print(f"{i}. {movie}")


# error handling
except:
    print("❌ Movie not found. Please check spelling or try another movie.")
