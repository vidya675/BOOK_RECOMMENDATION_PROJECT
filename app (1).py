import streamlit as st
import pickle as pkl
import pandas as pd
import random
st.title('Book recommandation System')


# Recreate the model here
movie_list = pkl.load(open('popular.pkl','rb'))
interactions_full_indexed_df = pkl.load(open('interactions_full_indexed_df.pkl', 'rb'))
interactions_train_indexed_df = pkl.load(open('interactions_train_indexed_df.pkl', 'rb'))
interactions_test_indexed_df = pkl.load(open('interactions_test_indexed_df.pkl', 'rb'))
books = pkl.load(open('books.pkl', 'rb'))
cf_preds_df = pkl.load(open('cf_preds_df.pkl', 'rb'))
ratings_explicit =  pkl.load(open('ratings_explicit.pkl', 'rb'))


User_ID = st.selectbox('User-ID',(movie_list))



class CFRecommender:
    MODEL_NAME = 'Collaborative Filtering'

    def __init__(self, cf_predictions_df):
        self.cf_predictions_df = cf_predictions_df

    def get_model_name(self):
        return self.MODEL_NAME

    def recommend_items(self, user_id, items_to_ignore=[], topn=10):
        # Get and sort the user's predictions
        sorted_user_predictions = self.cf_predictions_df[user_id].sort_values(ascending=False).reset_index().rename(
            columns={user_id: 'recStrength'})

        # Recommend the highest predicted rating content that the user hasn't seen yet.
        # remove for evaluation
        recommendations_df = sorted_user_predictions[
            ~sorted_user_predictions['ISBN'].isin(items_to_ignore)].sort_values('recStrength', ascending=False).head(
            topn)
        # recommendations_df = sorted_user_predictions.sort_values('recStrength', ascending = False).head(topn)
        recommendations_df = recommendations_df.merge(books, on='ISBN', how='inner')
        recommendations_df = recommendations_df[[ 'Book-Title']]

        return recommendations_df


cf_recommender_model = CFRecommender(cf_preds_df)


def get_items_interacted(UserID, interactions_df):
    interacted_items = interactions_df.loc[UserID]['ISBN']
    return set(interacted_items if type(interacted_items) == pd.Series else [interacted_items])


class ModelRecommender:

    # Function for getting the set of items which a user has not interacted with
    def get_not_interacted_items_sample(self, UserID, sample_size, seed=42):
        interacted_items = get_items_interacted(UserID, interactions_full_indexed_df)
        all_items = set(ratings_explicit['ISBN'])
        non_interacted_items = all_items - interacted_items
        # non_interacted_items = all_items

        random.seed(seed)
        non_interacted_items_sample = random.sample(non_interacted_items, sample_size)
        return set(non_interacted_items_sample)

    # Function to verify whether a particular item_id was present in the set of top N recommended items
    def _verify_hit_top_n(self, item_id, recommended_items, topn):
        try:
            index = next(i for i, c in enumerate(recommended_items) if c == item_id)
        except:
            index = -1
        hit = int(index in range(0, topn))
        return hit, index

    # Function to evaluate the performance of model for each user
    def evaluate_model_for_user(self, model, person_id):

        # Getting the items in test set
        interacted_values_testset = interactions_test_indexed_df.loc[person_id]

        if type(interacted_values_testset['ISBN']) == pd.Series:
            person_interacted_items_testset = set(interacted_values_testset['ISBN'])
        else:
            person_interacted_items_testset = set([int(interacted_values_testset['ISBN'])])

        interacted_items_count_testset = len(person_interacted_items_testset)

        # Getting a ranked recommendation list from the model for a given user
        person_recs_df = model.recommend_items(person_id, items_to_ignore=get_items_interacted(person_id,
                                                                                               interactions_train_indexed_df),
                                               topn=10000000000)
        print('Recommendation for User-ID = ', person_id)
        st.write(person_recs_df.head(10))

    def recommend_book(self, model, userid):

        person_metrics = self.evaluate_model_for_user(model, userid)
        return


model_recommender = ModelRecommender()

# Id_input = st.text_input('Enter the use ID')

if st.button("Click me"):
    output = (st.write(model_recommender.recommend_book(cf_recommender_model, User_ID)))
    