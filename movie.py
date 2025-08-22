import pandas as pd
import json
import matplotlib.pyplot as plt
import streamlit as st
import altair as alt
import numpy as np
from collections import Counter
import time
from itertools import cycle
from sklearn.manifold import TSNE
import pickle
import os


'''def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))'''


def parse_json(value):
    try:
        return json.loads(value)
    except (ValueError, SyntaxError, TypeError):
        return []

def extract_actor_names(cast_list):
    return [actor.get('name') for actor in cast_list if isinstance(actor, dict) and 'name' in actor]

class CreditsCsv:
    def __init__(self, file_path):
        self.file = file_path
        try:
            self.dataFrame = pd.read_csv(file_path, low_memory=False)

            if 'cast' in self.dataFrame.columns:
                self.dataFrame['cast'] = self.dataFrame['cast'].apply(parse_json)
                self.dataFrame['actor_names'] = self.dataFrame['cast'].apply(extract_actor_names)
            else:
                print("Warning: 'cast' column not found.")

        except FileNotFoundError:
            print(f"Error: The file '{file_path}' was not found.")
            self.dataFrame = pd.DataFrame()
        except pd.errors.EmptyDataError:
            print(f"Error: The file '{file_path}' is empty.")
            self.dataFrame = pd.DataFrame()
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            self.dataFrame = pd.DataFrame()

    def get_numeric_columns(self):
        return self.dataFrame.select_dtypes(include='number').columns.tolist()

    def get_categorical_columns(self):
        return self.dataFrame.select_dtypes(include='object').columns.tolist()

    def get_dataframe(self):
        return self.dataFrame
    
    def get_actors(self):
        return self.dataFrame.actor_names

    def get_movie_ids(self):
        return self.dataFrame.movie_id

    def get_titles(self):
        return self.dataFrame.title
    
#CreditsCsv

class MoviesCsv:
    def __init__(self, file_path):
        self.file = file_path
        try:
            self.data_frame = pd.read_csv(file_path, low_memory=False)
            self.data_frame['genres'] = self.data_frame['genres'].apply(parse_json)
            self.data_frame['production_companies'] = self.data_frame['production_companies'].apply(parse_json)
            self.data_frame['production_countries'] = self.data_frame['production_countries'].apply(parse_json)
            self.data_frame['spoken_languages'] = self.data_frame['spoken_languages'].apply(parse_json)


        except FileNotFoundError:
            print(f"Error: The file '{file_path}' was not found.")
            self.data_frame = pd.DataFrame()
        except pd.errors.EmptyDataError:
            print(f"Error: The file '{file_path}' is empty.")
            self.data_frame = pd.DataFrame()
        except (pd.errors.ParserError, UnicodeDecodeError) as e:
            print(f"Parsing error while reading the file: {e}")
            self.data_frame = pd.DataFrame()

    def get_numeric_columns(self):
        return self.data_frame.select_dtypes(include='number').columns.tolist()

    def get_categorical_columns(self):
        return self.data_frame.select_dtypes(include='object').columns.tolist()

    def get_dataframe(self):
        return self.data_frame
    
    def get_movie_ids(self):
        return self.data_frame.id

    def get_budget(self):
        return self.data_frame.budget

    def get_genres(self):
        return self.data_frame.genres

    def get_vote_averages(self):
        return self.data_frame.vote_average
    
    def get_release_date(self):
        return self.data_frame.release_date
    def get_title(self):
        return self.data_frame.title



#MoviesCsv

class Visualization():
    def __init__(self, credits, movies):
        self.credits = credits
        self.movies = movies

        self.actor_to_movies = {}
        self.movies_to_year = {}

        self.movie_id_row_credits = {}
        self.movie_id_row_movies = {}

        self.actor_to_movies = self.get_actor_to_movies()
        self.build_row_indices()

        COSTARRING_CHART_CACHE = 'costarringChartData.pkl'

        if os.path.exists(COSTARRING_CHART_CACHE):
            with open(COSTARRING_CHART_CACHE, 'rb') as f:
                self.costarring_chart_data = pickle.load(f)
        else:
            self.costarring_chart_data = self.compute_costarring_tsne_data()
            with open(COSTARRING_CHART_CACHE, 'wb') as f:
                pickle.dump(self.costarring_chart_data, f)  


    def get_actor_to_movies(self):
        actor_to_movies = {}
        for movie_id, actors in zip(self.credits.get_movie_ids(), self.credits.get_actors()):
            for actor in actors:
                actor_to_movies.setdefault(actor, []).append(movie_id)

        return actor_to_movies

    def build_row_indices(self):
        self.movie_id_row_credits = {
            movie_id: idx for idx, movie_id in enumerate(self.credits.get_movie_ids())
        }
        self.movie_id_row_movies = {
            movie_id: idx for idx, movie_id in enumerate(self.movies.get_movie_ids())
        }

    def get_actor_movies(self, actor_name):
        movie_ids = self.actor_to_movies.get(actor_name, [])
        rows = []

        df_movies = self.movies.get_dataframe()

        for movie_id in movie_ids:
            try:
                title = self.credits.get_title()[self.movie_id_row_credits[movie_id]]
                budget = self.movies.get_budget()[self.movie_id_row_movies[movie_id]]
                release_date = df_movies.loc[self.movie_id_row_movies[movie_id], "release_date"]
                year = None
                if pd.notnull(release_date):
                    year = str(release_date)[:4]

                rows.append({
                    "Film name": title,
                    "Budget": budget,
                    "Release Year": year

                })

            except (KeyError, IndexError):
                continue 

        return pd.DataFrame(rows)
    '''
    def information_about_actor(self, actorName):
        df = self.get_actor_movies(actorName)        

        st.metric("Max budget:", f"{df['Budget'].max():,}")
        st.subheader(f"Movies with {actorName}")
        st.dataframe(df)

        if not df.empty:
            st.subheader("Plot budget:")
            st.bar_chart(df.set_index("Film name"))'''

    def get_actor_list(self):
        return sorted(list(self.actor_to_movies.keys()))

    def multiple_actors_comparison_scatter(self, actors, show_fit=True):
        
        colVoteAverage = self.movies.get_vote_averages()
        colReleaseDate = self.movies.get_release_date()
        col_movie_name = self.movies.get_title()

        years, movie_score, actor_name, movie= zip(*[
            (colReleaseDate.iloc[movie_row],
             colVoteAverage.iloc[movie_row],
             actor, 
             col_movie_name.iloc[movie_row])
            for actor in actors
            for movie_row in map(lambda movie_id: self.movie_id_row_movies[movie_id], self.actor_to_movies[actor])            
        ])


        plot_data = pd.DataFrame({
            'year': years,
            'movie_score': movie_score ,
            'actor': actor_name,
            'movie': movie
        })
        base = alt.Chart(plot_data).mark_circle(
            size=100,
            opacity=1,
            stroke='black',
        ).encode(
            x = 'year',
            y = 'movie_score',
            color=alt.Color('actor', scale=alt.Scale(scheme='category10')),
            tooltip=['movie']
            
        ).interactive()
       
        st.altair_chart(base)

    def map_genres_to_films(self, actors):
        col_genres = self.movies.get_genres()
        colTitles = self.movies.get_title()
        genre_film_map = {}

        for actor in actors:
            genre_dict = {}
            movie_ids = self.actor_to_movies.get(actor, [])

            for movie_id in movie_ids:
                row_idx = self.movie_id_row_movies.get(movie_id)
                if row_idx is None:
                    continue

                genres = col_genres.iloc[row_idx]
                title = colTitles.iloc[row_idx]

                for genre in genres:
                    genre_name = genre.get('name')
                    if genre_name:
                        genre_dict.setdefault(genre_name, []).append(title)

            genre_film_map[actor] = genre_dict

        return genre_film_map

    def genre_distribution(self, actors):
        col_genres = self.movies.get_genres()
        genre_distributions = {}

        for actor in actors:
            genre_counter = Counter()
            movie_ids = self.actor_to_movies.get(actor, [])

            for movie_id in movie_ids:
                movie_row_id = self.movie_id_row_movies.get(movie_id)
                if movie_row_id is None:
                    continue

                genres = col_genres.iloc[movie_row_id]
                for genre in genres:
                    genre_name = genre.get('name')
                    if genre_name:
                        genre_counter[genre_name] += 1

            genre_distributions[actor] = genre_counter

        return genre_distributions

    def plot_individual_genre_rose_charts(self, actors):
        
        genre_distributions = self.genre_distribution(actors)
        

        for actor, genre_count in genre_distributions.items():
            if not genre_count:
                continue

            labels = list(genre_count.keys())
            counts = list(genre_count.values())

            angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)

            counts.append(counts[0])
            angles = np.append(angles, angles[0])
            labels.append(labels[0])

            short_labels = [label[:10] + '…' if len(label) > 10 else label for label in labels]

            fig, ax = plt.subplots(
                figsize=(3, 3),
                dpi=80,        
                subplot_kw={'projection': 'polar'}
            )
    
            ax.plot(angles, counts, 'o-', linewidth=1)
            ax.fill(angles, counts, alpha=0.2)

            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(short_labels[:-1], fontsize=6)  
            ax.set_yticklabels([])
            ax.set_rlim(0, max(counts) * 1.2)
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
            ax.tick_params(axis='x', pad=2) 

            plt.title(f"Genres for {actor}", fontsize=8, pad=15)
            plt.tight_layout() 

            st.pyplot(fig, use_container_width=False)

    def multiple_genres_table(self, actors):
        genre_film_mapping = self.map_genres_to_films(actors)

        for actor, genre_count in genre_film_mapping.items():
            if not genre_count:
                continue

            st.subheader(f"Information about {actor}")
            genre_film_rows = []
            all_films = set()

            for genre, films in genre_count.items():
                unique_films = sorted(set(films))
                all_films.update(unique_films)
                formatted_films = ", ".join(unique_films)
                genre_film_rows.append((genre, formatted_films))

            sorted_table = sorted(genre_film_rows, key=lambda x: len(x[1]), reverse=True)

            df = pd.DataFrame(sorted_table, columns=["Genre", "Film names"])

            styled = df.style.set_properties(subset=["Genre"], **{
                'white-space': 'nowrap',
                'text-align': 'left'
            }) \
            .set_properties(subset=["Film names"], **{
                'white-space': 'normal',  
                'text-align': 'left'
            })


            st.table(styled)
            st.markdown(f"**Total films:** {len(all_films)}")


    def plot_combined_genre_rose_chart(self, actors):
        genre_distributions = self.genre_distribution(actors)

        all_genres = sorted(set().union(*[d.keys() for d in genre_distributions.values()]))

        if not all_genres:
            st.warning("No genre data available for the selected actors.")
            return

        genre_to_index = {genre: i for i, genre in enumerate(all_genres)}
        num_genres = len(all_genres)
        angles = np.linspace(0, 2 * np.pi, num_genres, endpoint=False)
        angles = np.append(angles, angles[0])  

        fig, ax = plt.subplots(figsize=(5, 5), dpi=100, subplot_kw={'projection': 'polar'})

        color_cycle = cycle(plt.cm.tab10.colors)
        max_value = 0

        for actor in actors:
            genre_count = genre_distributions.get(actor, {})
            values = [genre_count.get(genre, 0) for genre in all_genres]
            values.append(values[0])  

            color = next(color_cycle)
            ax.plot(angles, values, label=actor, linewidth=2, marker='o', color=color)
            ax.fill(angles, values, alpha=0.1, color=color)

            max_value = max(max_value, max(values)) if values else max_value

        short_labels = [g[:10] + '…' if len(g) > 10 else g for g in all_genres]
        short_labels.append(short_labels[0])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(short_labels[:-1], fontsize=8)
        ax.set_yticklabels([])
        ax.set_rlim(0, max_value * 1.2 if max_value > 0 else 1)
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.tick_params(axis='x', pad=2)

        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)
        plt.title("Genre distribution for selected actors", fontsize=10, pad=20)
        plt.tight_layout()

        st.pyplot(fig)
    
    def plot_budget_vs_rating(self):
        movie_score = self.movies.get_vote_averages().tolist()
        movie_budget = self.movies.get_budget().mask(lambda x: x <= 0, 1).tolist()
        title = self.movies.get_title().astype(str).tolist()

        plot_data = pd.DataFrame({
            'score': movie_score,
            'budget': movie_budget 
        })
        chart = alt.Chart(plot_data).mark_circle(size=100).encode(
            x = alt.X('budget', scale=alt.Scale(type='log')),
            y = 'score'
        ).interactive()

        st.altair_chart(chart, use_container_width=True)

    def compute_costarring_tsne_data(self):
        MOST_POPULAR_ACTORS_LIMIT = 500

        actors_id = {actor: i for i, actor in enumerate(self.actor_to_movies)}
        ids_actor = {i: actor for i, actor in enumerate(self.actor_to_movies)}

        list_actor_id_movies = []
        for actor, movies in self.actor_to_movies.items():
            actor_id = actors_id[actor]
            list_actor_id_movies.append((actor_id, len(movies)))

        list_actor_id_movies = sorted(list_actor_id_movies, key=lambda aid_nm: -aid_nm[1])

        most_popular_actors_ids = [aid_nm[0] for aid_nm in list_actor_id_movies[:MOST_POPULAR_ACTORS_LIMIT]]
        most_popular_actors_names = [ids_actor[aId] for aId in most_popular_actors_ids]

        actor_id_to_most_popular_actor_id = {actorId: i for i, actorId in enumerate(most_popular_actors_ids)}

        actors_coupling = np.zeros((MOST_POPULAR_ACTORS_LIMIT, MOST_POPULAR_ACTORS_LIMIT))

        for actors_in_the_movie in self.credits.get_actors():
            popular_actors_in_the_movie = []
            for actor in actors_in_the_movie:
                actorId = actors_id[actor]
                if actorId in actor_id_to_most_popular_actor_id:
                    mpaId = actor_id_to_most_popular_actor_id[actorId]
                    popular_actors_in_the_movie.append(mpaId)

            for i in popular_actors_in_the_movie:
                for j in popular_actors_in_the_movie:
                    actors_coupling[i, j] += 1
                    actors_coupling[j, i] += 1

        print("start computing mds")
        time_start = time.time()

        actors_distance = np.vectorize(lambda nCommonMovies: 1.0 / (1 + nCommonMovies) ** 2)(actors_coupling)

        tsne = TSNE(metric="precomputed", perplexity=4, n_components=2, init='random', random_state=42)
        actors_coords_2d = tsne.fit_transform(actors_distance)

        ime_end = time.time()
        print("Elapsed time:", time_end - time_start, "seconds")

        plot_data = pd.DataFrame({
            'x': actors_coords_2d[:, 0],
            'y': actors_coords_2d[:, 1],
            'actor': most_popular_actors_names
        })

        chart = alt.Chart(plot_data).mark_circle(size=100).encode(
            x='x',
            y='y',
            color="actor"
        ).interactive()

        return plot_data

    def build_costarring_chart(self, data):
        chart = alt.Chart(data).mark_circle(size=100).encode(
            x='x',
            y='y',
            color=alt.Color('actor', scale=alt.Scale(scheme='category10'), legend=None), 
            tooltip=['actor']
        ).interactive()


        return chart

    def show_costarring_chart(self, selected_actor=None):
        REGION_SIZE = 5

        chart = self.build_costarring_chart(self.costarring_chart_data)

        if selected_actor:
            actor_names = self.costarring_chart_data['actor'].values

            if selected_actor in actor_names:
                selected_id = np.where(actor_names == selected_actor)[0][0]
                x = float(self.costarring_chart_data['x'].iloc[selected_id])
                y = float(self.costarring_chart_data['y'].iloc[selected_id])

                min_x = x - REGION_SIZE
                max_x = x + REGION_SIZE
                min_y = y - REGION_SIZE
                max_y = y + REGION_SIZE

                chart = chart.encode(
                    x=alt.X('x', scale=alt.Scale(domain=[min_x, max_x])),
                    y=alt.Y('y', scale=alt.Scale(domain=[min_y, max_y]))
                )

        st.altair_chart(chart, use_container_width=True)


    def show_costars_for_actor(self, selected_actor_name):
        costarring = {}
        movie_ids = self.actor_to_movies[selected_actor_name]
        for movie_id in movie_ids:
            i_row_credits = self.movie_id_row_credits[movie_id]
            cast = self.credits.get_actors()[i_row_credits]
            title = self.credits.get_titles()[i_row_credits]
            
            for actor in cast:
                if actor == selected_actor_name: continue
                costarring.setdefault(actor, []).append(title)
        costarring = sorted(costarring.items(), key = lambda actor_movies: -len(actor_movies[1]))
        for i in range(min(10, len(costarring))):
            formatted_films = ", ".join(sorted(set(costarring[i][1])))
            st.subheader(f"{costarring[i][0]}: {formatted_films}")


#Visualization