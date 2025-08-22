#python -m streamlit run streamlit_app.py

from movie import CreditsCsv, MoviesCsv
from movie import Visualization
import streamlit as st


def main():

    st.set_page_config(
        page_title="pyMDB",
        layout="wide"
    )    

    CREDITS_CSV_PATH = "tmdb_5000_credits.csv"
    MOVIES_CSV_PATH = "tmdb_5000_movies.csv"
    credits_file = CreditsCsv(CREDITS_CSV_PATH)
    movies_file = MoviesCsv(MOVIES_CSV_PATH)
    stats = Visualization(credits_file, movies_file)

    comparisonActors, GenresRose, GenresRoseMultiple, BudgetVSscore, tabCostarring = st.tabs([
        'Comparison of actors',
        'Rose of genres for one actor',
        'Rose of genres for many actors',
        'Budget VS score',
        'Costarring'
    ])

    with comparisonActors:
        st.title("Actor statistics")
        col1, col2 = st.columns([1, 3])

        with col1:
            actors = st.multiselect("Choose an actor", stats.get_actor_list())
        
        with col2:
            if actors:
                stats.multiple_actors_comparison_scatter(actors)

    with GenresRose:
        st.title("In which genres the actor was starring")
        col9, col10 = st.columns([1, 3])
    
        with col9:
            selected_actor = st.selectbox(
                "Choose an actor", 
                stats.get_actor_list(), 
                key='Rose of genres for one actor'
            )
        
        with col10:
            if selected_actor:
                with st.expander(f"Actor: {selected_actor}", expanded=True):
                    rose_col1, data_col1 = st.columns([2, 3])

                    with rose_col1:
                        stats.plot_individual_genre_rose_charts([selected_actor])

                    with data_col1:
                        stats.multiple_genres_table([selected_actor])

    with GenresRoseMultiple:
        st.title("In which genres the actors were starring")
        col3, col4 = st.columns([1, 3])

        with col3:
            selected_actors = st.multiselect(
                "Choose actors", 
                stats.get_actor_list(), 
                key='Rose of genres for many actors'
            )

        with col4:
            if selected_actors:
                with st.expander("Selected actors", expanded=True):
                    rose_col, data_col = st.columns([2, 3])

                    with rose_col:
                        stats.plot_combined_genre_rose_chart(selected_actors)

                    with data_col:
                        stats.multiple_genres_table(selected_actors)

    with BudgetVSscore:
        st.title("Correlation between budget and score")
        col5, col6 = st.columns([3, 1])
    
        with col5:
            stats.plot_budget_vs_rating()

    with tabCostarring:
        st.title("Costarring")
        col7, col8 = st.columns([1,3])
        with col7:
            actor = st.selectbox("Choose an actor", stats.get_actor_list())
        
        with col8:
            stats.show_costarring_chart (actor)
            if actor:
                stats.show_costars_for_actor(actor)


    




if __name__ == "__main__":
    main()
