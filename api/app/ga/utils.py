"""
Module for Miscellaneous functions.
"""

import random
import pandas as pd
from typing import List
import plotly.graph_objects as go

class Node:
    # Each node in the heap has a weight, value, and total weight.
    # The total weight, self.tw, is self.w plus the weight of any children.
    __slots__ = ['w', 'v', 'tw']
    def __init__(self, w, v, tw):
        self.w, self.v, self.tw = w, v, tw

def rws_heap(items):
    # h is the heap. It's like a binary tree that lives in an array.
    # It has a Node for each pair in `items`. h[1] is the root. Each
    # other Node h[i] has a parent at h[i>>1]. Each node has up to 2
    # children, h[i<<1] and h[(i<<1)+1].  To get this nice simple
    # arithmetic, we have to leave h[0] vacant.
    h = [None]                          # leave h[0] vacant
    for w, v in items:
        h.append(Node(w, v, w))
    for i in range(len(h) - 1, 1, -1):  # total up the tws
        h[i>>1].tw += h[i].tw           # add h[i]'s total to its parent
    return h

def rws_heap_pop(h):
    gas = h[1].tw * random.random()     # start with a random amount of gas

    i = 1                     # start driving at the root
    while gas >= h[i].w:      # while we have enough gas to get past node i:
        gas -= h[i].w         #   drive past node i
        i <<= 1               #   move to first child
        if gas >= h[i].tw:    #   if we have enough gas:
            gas -= h[i].tw    #     drive past first child and descendants
            i += 1            #     move to second child
    w = h[i].w                # out of gas! h[i] is the selected node.
    v = h[i].v

    h[i].w = 0                # make sure this node isn't chosen again
    while i:                  # fix up total weights
        h[i].tw -= w
        i >>= 1
    return v

def random_weighted_sample_no_replacement(items, n):
    heap = rws_heap(items)              # just make a heap...
    for i in range(n):
        yield rws_heap_pop(heap)        # and pop n items off it.

def get_season_games(season: int = 2022) -> pd.DataFrame:
    games = pd.read_csv(f'./ga/data/training_data_{season}.csv')
    games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE'])
    games = games.sort_values(by=['GAME_DATE'])

    return games

def get_games_by_month(season: int = 2023) -> List[pd.DataFrame]:
    games = get_season_games(season)

    dfs_by_month_year = []
    grouped = games.groupby([games['GAME_DATE'].dt.year, games['GAME_DATE'].dt.month])

    for _, group in grouped:
        small_df = group.copy()
        dfs_by_month_year.append(small_df)

    return dfs_by_month_year

def multiple_line_charts(data_list, title, x_label, y_label, labels=None):
    fig = go.Figure()
    fig.update_layout(title=title, xaxis_title=x_label, yaxis_title=y_label)

    if labels is None:
        labels = [f"Line {i + 1}" for i in range(len(data_list))]

    for data, label in zip(data_list, labels):
        fig.add_trace(go.Scatter(x=list(range(1, len(data) + 1)), y=data, mode='lines', name=label))

    return fig
