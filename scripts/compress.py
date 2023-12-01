
import requests
import json

def get_chessdotcom_pgns_from_date(username: str, month: str, year: str) -> str:
    """Gets the PGNs from the given Chess.com username and date

    :param username: Chess.com username
    :type username: str
    :param month: Month to get the PGNs from
    :type month: str
    :param year: Year to get the PGNs from
    :type year: str
    :return: The PGNs from the given Chess.com username and date
    :rtype: str
    """
    api_url = f"https://api.chess.com/pub/player/{username}/games/{year}/{month}"
    games = json.loads(requests.get(api_url).text)["games"]

    pgns = []
    for game in games:
        pgns.append(game["pgn"])

    return pgns

def get_all_chessdotcom_pgns(username: str) -> list:
    """Gets every game's PGN from the given Chess.com username

    :param username: Chess.com username
    :type username: str
    :return: List of PGNs from every Chess.com game
    :rtype: list of str
    """
    api_url = f"https://api.chess.com/pub/player/{username}/games/archives"
    games_archive = json.loads(requests.get(api_url).text)["archives"]

    pgns = []
    for endpoint in games_archive:
        games = json.loads(requests.get(endpoint).text)["games"]
        for game in games:
            pgns.append(game["pgn"])
    
    return pgns