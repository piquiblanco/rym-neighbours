import os, csv
from datetime import datetime
import pandas as pd
import numpy as np
import json


class Neighbourhood:
    def __init__(self):
        with open("userdata/rym_friends.csv", newline="") as f:
            reader = csv.reader(f)
            self.rym_friends = [row[0] for row in list(reader)]
        self.album_map = json.load(open('album_map.json', encoding='utf-8'))
        self.stars = {
            "5.00 stars": 5.0,
            "4.50 stars": 4.5,
            "4.00 stars": 4.0,
            "3.50 stars": 3.5,
            "3.00 stars": 3.0,
            "2.50 stars": 2.5,
            "2.00 stars": 2.0,
            "1.50 stars": 1.5,
            "1.00 stars": 1.0,
            "0.50 stars": 0.5,
        }
        self.album_translator = {}
        self.userfiles = {}
        self.albums = set()
        my_ratings = pd.read_excel(
            "FASTCAR.xlsm", index_col=0, header=2, usecols=[0, 1, 2, 17]
        )
        my_ratings["album_name"] = my_ratings["Artist"] + " - " + my_ratings["Album"]
        my_ratings["SCORE_n"] = (
            my_ratings["SCORE"] - my_ratings["SCORE"].mean()
        ) / my_ratings["SCORE"].std()
        my_ratings.replace(self.album_map, inplace=True)
        self.my_ratings = my_ratings[["album_name", "SCORE", "SCORE_n"]].copy()
        self.ratings = {}
        self.rating_counts = None

    def check_current_friends(self):
        usernames_current = []
        for filename in os.listdir("userdata/current"):
            usernames_current.append(filename.split(".")[0])
        only_files = set(usernames_current).difference(set(self.rym_friends))
        only_list = set(self.rym_friends).difference(set(usernames_current))
        if only_files:
            print("Usernames not found on friends list:")
            for username in only_files:
                # print(username)
                pass
        if only_list:
            print("Usernames not found in current files:")
            for username in only_list:
                print(username)
        # todo: fix differences if any? which should be the source of truth: list, files, or the sum of the two?

    def load_new_files(self):
        today = datetime.today().strftime("%Y-%m-%d")
        today_dir = "userdata/discarded/" + today
        if not os.path.exists(today_dir):
            os.mkdir(today_dir)
        # three folders: new, current, discarded
        # for all new files:
        #   if user already there, move new file to current, move old file to discarded
        #   if user is new, move new file to current, add user to list of new users
        for filename in os.listdir("userdata/new"):
            replace = False
            if os.path.exists("userdata/current/" + filename):
                os.rename("userdata/current/" + filename, today_dir + "/" + filename)
                replace = True
            os.rename("userdata/new/" + filename, "userdata/current/" + "/" + filename)
            if replace:
                print(filename + " replaced.")
            else:
                print(filename + " added.")
        # todo still: add new users to list of new users

    def is_number(self, s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    def read_files(self):
        for filename in os.listdir("userdata/current"):
            print(filename)
            df = pd.read_csv("userdata/current/" + filename, header=0, sep="\t")
            df[df.columns[0]].replace(self.stars, inplace=True)
            df[df.columns[2]].replace(self.stars, inplace=True)
            df = df[list(map(self.is_number, df[df.columns[0]]))]
            df = df.apply(pd.to_numeric, errors="ignore")
            self.userfiles[df.columns[2]] = df
            user_albums = df["Album"].tolist()
            self.albums.update(user_albums)

        all_results = pd.DataFrame(data=list(self.albums), columns=["album"])
        all_results.set_index("album", inplace=True)

        for username in self.userfiles:
            all_results.insert(
                loc=len(all_results.columns), column=username, value=np.nan
            )
            for index, row in self.userfiles[username].iterrows():
                rental, album, user = row
                all_results.loc[album, username] = user
        self.ratings["real"] = all_results
        self.ratings["normalized"] = self.normalize_ratings(all_results)
        self.rating_counts = self.ratings["normalized"].count()

    def normalize_ratings(self, ratings):
        r2 = ratings.copy()
        r2.loc[:, r2.columns != "album"] = (
            r2.loc[:, r2.columns != "album"] - r2.mean()
        ) / r2.std()
        return r2

    def predict_ratings(self, group):
        selected_ratings = self.ratings["normalized"][group]
        df = selected_ratings.mean(axis=1)
        return pd.DataFrame({"rating": df.fillna(-1)})

    def assess(self, prediction):
        comparison_df = pd.merge(
            prediction,
            self.my_ratings,
            how="inner",
            left_index=True,
            right_on="album_name",
        )
        comparison_df = comparison_df[["album_name", "rating", "SCORE_n"]]
        error = ((comparison_df["rating"] - comparison_df["SCORE_n"]) ** 2).mean()
        return error

    def assess_candidates(self, chosen, candidates):
        assessments = {}
        for user in candidates:
            group = list(chosen.keys())
            group.append(user)
            prediction = self.predict_ratings(group)
            error = self.assess(prediction)
            ratings = self.rating_counts.loc[user]
            special_albums = {
                "Women - Public Strain": 10,
                "Institute - Catharsis": 20,
                "Cindy Lee - Act of Tenderness": 20,
                "Parquet Courts - Light Up Gold": 30,
                "Swans - The Seer": -20
            }
            result = error - (ratings ** 0.5) / 200
            for key in special_albums.keys():
                denominator = special_albums[key]
                rating = self.ratings["normalized"].loc[key].loc[user]
                if not np.isnan(rating):
                    result -= rating / denominator
            assessments[user] = result
        return pd.DataFrame(assessments, index=[0]).transpose()

    def neighbours_ranking(self):
        chosen = {}
        candidates = list(self.ratings["real"].columns)
        while candidates:
            assessments = self.assess_candidates(chosen, candidates)
            best = assessments.idxmin(0)[0]
            candidates.remove(best)
            chosen[best] = round(assessments.loc[best][0], 2)
            print(best)
        return chosen

    def find_differences(self):
        my_albums = set(self.my_ratings["album_name"])
        their_albums = set(self.ratings["normalized"].index)
        pd.Series(list(my_albums.difference(their_albums))).to_excel("my_albums.xlsx")
        pd.Series(list(their_albums.difference(my_albums))).to_excel(
            "their_albums.xlsx"
        )


def prepare_all_data():
    # for file in current files:
    # read_file, append to master data frame
    # OR
    # add to dictionary with usernames as keys, including my username
    pass


def evaluate():
    # prepare list of users to evaluate
    # while the list is not empty:
    #   evaluate each user's input, pick best user
    #   add him to ordered list, with evaluation of his input
    # return ordered list of users
    # also, return suggestions to add/remove users from friends
    # take parameters: prior average (avg/fixed), use prior (yes/no), public strain?
    # cutoff (possibly point between positive and negative influence)
    # loss function: sum of squares? maybe higher rated albums should have even higher impact?
    # read y/n input to accept changes on new/current/discarded userlists, save a file with links to add/remove users!!
    # also, save a file with unmatched albums from the site
    pass


def read_file_from_html():
    # read not .txt but .html!
    pass


def reshape_data():
    # long to wide: each user is a variable
    pass


def regression():
    # run linear regression on normalized data, with nulls replaced by zeros
    # return list of coefficients, matched albums, and p-values for users
    # save model object
    pass


def predict_albums():
    # using the model object
    # predict ratings for all albums, compare them to real ratings, sort by residual
    pass


def prepare_data_for_tableau():
    # prepare data in tableau-comprehensible format.
    pass


nb = Neighbourhood()
nb.read_files()
dupa = nb.neighbours_ranking()
print(dupa)

#nb.find_differences()
