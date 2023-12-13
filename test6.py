import functions as fs


notes_to_play = ['B4', 'B4', 'A4', 'G4', 'A4', 'B4', 'B4', 'A4', 'A4', 'A4', 'D5']
temp_dict = {}
for note in notes_to_play:
    positions = fs.get_number_upgrade(note)
    temp_dict[note] = positions
ans = fs.calculate_efficient_positions(notes_to_play, temp_dict)
print(ans)