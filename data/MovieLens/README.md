We sampled about 10M interactions based on movielens-latest, and crawled the corresponding item side information from tmdb and movielens.
We first tried to crawl from tmdb, and some the them is not found, and then we searched the title corresponding to the movieId, so that we get the remaped tmdbId.
But still, there are some movies lost in tmdb, we crawled the information from movielens.

You can download files below with https://rec.ustc.edu.cn/share/c6189690-a562-11ee-8c5d-fb78b74f9bd1

- `sampled_items.jsonl` is the item side information.
- `sampled_ratings.csv` is the sampled interactions.
- `sampled_summaries.json` is the organized text for each item.