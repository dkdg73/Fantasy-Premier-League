#%%

from parsers import *
from cleaners import *
from getters import *
from collector import collect_gw, merge_gw
from understat import parse_epl_data
import csv

def parse_data(season, gw = None):
    """ 
    Parse and store all the data for a given gw
    Default gw will be latest
    Input gw to alter
    """
    if gw != None:
        if type(gw) != int:
            Exception("gw must be integer")
    season = season
    base_filename = 'data/' + season + '/'
    print("Getting data")
    data = get_data()
    print("Parsing summary data")
    parse_players(data["elements"], base_filename) #write 'elements' to players_raw.csv
    xPoints = []
    for e in data["elements"]:
        xPoint = {}
        xPoint['id'] = e['id']
        xPoint['xP'] = e['ep_this'] # ep_this has null value until GW1 begins, by which time we can't submit a team
        #xPoint['xP'] = e['ep_next']
        xPoints += [xPoint]
    if gw == None:
        gw_num = 0
        events = data["events"]
        for event in events:
            if event["is_current"] == True:
                gw_num = event["id"]
    else:
        gw_num = gw
    print("Cleaning summary data")
    clean_players(base_filename + 'players_raw.csv', base_filename) #write updated summary stats for players to cleaned_players.csv
    print("Getting fixtures data")
    fixtures(base_filename) #write updated fixtures data to fixtures.csv
    print("Getting teams data")
    parse_team_data(data["teams"], base_filename) 
    print("Extracting player ids")
    id_players(base_filename + 'players_raw.csv', base_filename)
    player_ids = get_player_ids(base_filename)
    num_players = len(data["elements"])
    player_base_filename = base_filename + 'players/'
    gw_base_filename = base_filename + 'gws/'
    print("Extracting player specific data")
    for i,name in player_ids.items():
        player_data = get_individual_player_data(i)
        parse_player_history(player_data["history_past"], player_base_filename, name, i)
        parse_player_gw_history(player_data["history"], player_base_filename, name, i)
    if gw_num > 0:
        print("Writing expected points")
        os.makedirs(gw_base_filename, exist_ok=True)
        with open(os.path.join(gw_base_filename, 'xP' + str(gw_num) + '.csv'), 'w+', newline='') as outf:
            w = csv.DictWriter(outf, ['id', 'xP'])
            w.writeheader()
            for xp in xPoints:
                w.writerow(xp)
        print("Collecting gw scores")
        collect_gw(gw_num, player_base_filename, gw_base_filename, base_filename) 
        print("Merging gw scores")
        merge_gw(gw_num, gw_base_filename)
    understat_filename = base_filename + 'understat'
    parse_epl_data(understat_filename)

def fixtures(base_filename):
    data = get_fixtures_data()
    parse_fixtures(data, base_filename)
    
#%%
def main():
    parse_data('2024-25', gw = 1)

if __name__ == "__main__":
    main()


# %%
