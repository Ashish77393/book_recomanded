from django.shortcuts import render
import pickle
import requests
from sklearn.neighbors import NearestNeighbors
import numpy as np

# Load pre-saved data
final_rating = pickle.load(open('final_rating.pkl', 'rb'))
book_pivot = pickle.load(open('book_pivot.pkl', 'rb'))

# Initialize the model and fit it
model = NearestNeighbors(algorithm='brute')
model.fit(book_pivot)  # Ensure the model is fitted with book_pivot

def view(request):
    title = list(final_rating['title'].values)
    author = list(final_rating['author'].values)
    img = list(final_rating['Image-URL-S'].values)
    rating = list(final_rating['rating'].values)

    data = {
        'items': zip(title, author, img, rating)  
    }
    return render(request, 'view.html', data)

def about(request):
    return render(request, 'about.html')

def booksugg(request):
    return render(request, 'booksugg.html')

def recommanded(request):
    user_input = request.POST.get('user_input')  # Changed this line
    if user_input:
        try:
            bookid = np.where(book_pivot.index == user_input)[0][0]
        except IndexError:
            # Handle case where book is not found
            return render(request, 'booksugg.html', {'error': 'Book not found'})
        
        distances, suggestions = model.kneighbors(book_pivot.iloc[bookid, :].values.reshape(1, -1), n_neighbors=6)
        data = []
        for i in range(len(suggestions[0])): 
            item = []
            temp_df = final_rating[final_rating['title'] == book_pivot.index[suggestions[0][i]]]
            item.extend(list(temp_df.drop_duplicates('title')['title'].values))
            item.extend(list(temp_df.drop_duplicates('title')['author'].values))
            item.extend(list(temp_df.drop_duplicates('title')['Image-URL-S'].values))
            data.append(item)
        
        return render(request, 'booksugg.html', {'data': data})
    
    return render(request, 'booksugg.html')
