import numpy as np
import pandas as pd


def get_famtype(df):
    df["FamSize"] = df["SibSp"] + df["Parch"] + 1
    df["FamType"] = pd.cut(
        df["FamSize"],
        bins=[0, 1, 4, 12],
        labels=["Solo", "Small", "Large"]
    )
    
    return df

def extract_title(df):
    df["Title"] = df["Name"].str.extract(r",\s*([^\.]+)\.")
    # df["Title"] = df["Title"].replace(
    #     [
    #         "Lady", "the Countess", "Capt", "Col", "Don", "Dr",
    #         "Major", "Rev", "Sir", "Jonkheer", "Dona"
    #     ],
    #     "Rare"
    # )
    
    df["Title"] = df["Title"].replace(
        {"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"}
    )

    df["Title"] = df["Title"].replace(
        [
            "Capt", "Col", "Major", "Dr", "Rev"
        ],
        "Officer"
    )
    df["Title"] = df["Title"].replace(
        [
            "Jonkheer", "Don", "Sir", "the Countess", "Dona", "Lady"
        ],
        "Royalty"
    )
    
    df.drop("Name", axis=1, inplace=True)
    
    return df

def extract_deck(df, cabin_col="Cabin"):
    """
    Extracts deck letter from the Cabin column.
    Missing or NaN cabins are labeled as 'Unknown'.
    """
    
    # Extract first letter if not null, else 'Unknown'
    df["Deck"] = df[cabin_col].apply(lambda x: str(x)[0] if pd.notnull(x) else "Unknown")
    # df.drop("Cabin", axis=1, inplace=True)
    
    return df

def bin_fare(df):
    bins = [-1, 0, 50, 100, 200, 300, float("inf")]
    labels=["Zero", "Very Low", "Low", "Medium", "High", "Very High"]

    df["FareBinned"] = pd.cut(
        df["Fare"],
        bins=bins,
        labels=labels
    )
    return df

def map_ticket(df, threshold=3):
    ticket_counts = df["Ticket"].value_counts()
    df["TicketGroupSize"] = df["Ticket"].apply(lambda x: x if ticket_counts[x] > threshold else "OTHER")
    return df

def extract_num_cabins(df, cabin_col="Cabin"):
    """
    Creates a new column 'NumCabins' counting how many cabins a passenger had.
    """
    df = df.copy()
    
    # NaN → 0 cabins, otherwise count number of cabins (split by space)
    df["NumCabins"] = df[cabin_col].fillna("").apply(lambda x: len(x.split()) if x else 0)
    
    return df
