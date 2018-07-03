import numpy as np

def find_discord(mp, window_size, k=3):
    sort_mp = np.sort(mp)[::-1]
    sort_mp_idx = np.argsort(mp)[::-1]

    discord_idx = np.ones((k, 1)) * -1
    for i in range(k):
        if len(sort_mp) < i:
            break

        discord_idx.append(sort_mp[0])
        sort_mp = np.delete(sort_mp, 0)

        exc_zone = np.around(window_size / 2.0)
        neighbor = np.where(np.abs(sort_mp_idx - discord_idx[0]) < exc_zone)[0]
        sort_mp = np.delete(sort_mp, neighbor)

    discord_idx = np.delete(discord_idx, np.where(discord_idx == -1)[0])
    return discord_idx
