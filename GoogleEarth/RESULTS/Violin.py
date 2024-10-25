
import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np
import pandas as pd
# For Dataset DATSETROT# 


estim_pixels_DATSETROT = [
    (425697.392425685, 907153.221936464),
    (425402.416440415, 907237.813217097),
    (425066.719049313, 907595.465642666),
    (424937.030327420, 907957.529999368),
    (425111.796165922, 908158.382187556),
    (425469.871602842, 908698.355939215),
    (425391.762755027, 909072.428458352),
    (425208.068389433, 909410.002629235),
    (425045.232205221, 909533.447292253),
    (424719.557260604, 909510.990766533),
    (424415.683996410, 909484.965319743),
    (424095.401552683, 909586.803872526),
    (423821.618040653, 909826.413518630)
]

actual_pixels_DATSETROT = [
    (425694.424664380, 907136.188588694),
    (425406.675811371, 907254.720173363),
    (425044.442643582, 907594.931831427),
    (424917.915270221, 907974.678789205),
    (425127.380018784, 908140.474378795),
    (425483.162605801, 908684.976523538),
    (425383.010730664, 909064.574852370),
    (425206.697753711, 909409.691265662),
    (425047.389274582, 909533.796435816),
    (424714.645228278, 909523.169466156),
    (424408.202066949, 909499.017262383),
    (424095.204667145, 909598.747285346),
    (423829.818241520, 909833.209448123)
]


estim_pixels_DATSETCPT = [
    (545642.425307973, 1181023.490254589),
    (545526.568594421, 1181372.278472564),
    (544917.600666893, 1181564.655789142),
    (544560.435692500, 1181767.159852005),
    (544309.107656883, 1182052.085980182),
    (543943.479108107, 1182272.637867992),
    (543647.311662095, 1182564.714397414),
    (543995.468694092, 1182936.133508207),
    (544353.259989550, 1183149.877084598),
    (544757.364319376, 1183161.724826444),
    (545074.304871591, 1183195.159331188),
    (544792.178969015, 1183615.484836518),
    (544536.988852029, 1183929.746302092),
    (544208.898126811, 1184178.857883392)
]

actual_pixels_DATSETCPT = [
    (545652.576791282, 1181025.168304107),
    (545521.515217650, 1181369.406514355),
    (544917.595422331, 1181567.220093704),
    (544561.320203728, 1181765.033673053),
    (544308.470218263, 1182051.302052562),
    (543942.057750772, 1182274.955272658),
    (543649.404147047, 1182566.159313882),
    (543995.301171201, 1182936.817830961),
    (544354.173867294, 1183150.793282987),
    (544757.987048946, 1183161.245272502),
    (545072.954516133, 1183197.827235807),
    (544795.561645204, 1183616.487482512),
    (544533.896192641, 1183929.369724213),
    (544209.383955771, 1184176.152809996)
]


# For Dataset DATSETROCK
estim_pixels_DATSETROCK = [
    (647865.239661762, 1152286.107510492),
    (648256.709499911, 1152075.486068395),
    (647809.185672753, 1152357.380485204),
    (647692.635278998, 1153092.423029827),
    (647368.384678496, 1153007.792592071),
    (647699.521297053, 1152666.466859102),
    (647110.921517168, 1152834.891940044),
    (646252.639306544, 1152598.990612504),
    (646221.156545453, 1152604.245299942),
    (645366.076405567, 1152526.524886128),
    (644620.053696962, 1152849.069306983),
    (644246.824015579, 1153414.548697392),
    (644176.063179020, 1153577.831822046)
]

actual_pixels_DATSETROCK = [
    (647864.824914293, 1152285.550613073),
    (648250.295286179, 1152075.069175329),
    (647809.683033979, 1152357.976515024),
    (647683.688921436, 1153095.523760244),
    (647361.243264643, 1153022.692110943),
    (647696.078844507, 1152674.662321595),
    (647116.485165296, 1152821.847172759),
    (646252.362558771, 1152599.700498707),
    (646221.431240220, 1152603.555098531),
    (645373.144712779, 1152517.739534034),
    (644615.093595218, 1152842.134540252),
    (644253.513485238, 1153411.296635278),
    (644169.456715344, 1153581.000464360)
]


# For Dataset DATSETSAND
estim_pixels_DATSETSAND = [
    (-159928.089327469, -709271.371959472),
    (-159867.961247237, -709035.825865604),
    (-159978.015188987, -708840.206408913),
    (-160156.670907484, -708873.269185220),
    (-160380.058791928, -708650.380156137),
    (-160513.703952856, -708358.100228977),
    (-160075.104774316, -707840.452151956),
    (-159745.139893610, -707710.986891681),
    (-159653.161964289, -707503.620741172),
    (-159872.255691470, -707327.290971785),
    (-159770.385860966, -707018.656699465),
    (-159476.958671626, -706757.852176497),
    (-159588.678248670, -706386.772032646)
]

actual_pixels_DATSETSAND = [
    (-159942.394805857, -709270.399124423),
    (-159855.196964270, -709036.379171198),
    (-159971.270829942, -708840.828251380),
    (-160161.366594356, -708868.578048915),
    (-160367.181108696, -708642.772837199),
    (-160504.956288213, -708359.364255147),
    (-160067.670432321, -707848.247045663),
    (-159752.557678915, -707704.589249034),
    (-159660.059046599, -707508.737789893),
    (-159865.866269192, -707322.703948652),
    (-159778.254987309, -707013.048265682),
    (-159471.195848385, -706766.205301321),
    (-159568.395296443, -706379.611551537)
]



# For Dataset DATSETAMAZ
estim_pixels_DATSETAMAZ = [
    (-2009794.425553445, 111712.251483730),
    (-2010128.759913699, 111609.827129832),
    (-2010197.271492821, 111357.914446429),
    (-2010253.236780631, 110567.379417169),
    (-2009308.201197505, 110596.584175627),
    (-2009122.402337395, 110734.774072588),
    (-2009045.688271298, 110809.739330295),
    (-2009030.509370577, 111117.525335034),
    (-2009299.556710538, 111167.194594686),
    (-2009654.248535948, 111340.228427699)
]

actual_pixels_DATSETAMAZ = [
    (-2009789.987877973, 111709.917451048),
    (-2010132.598215094, 111629.728122077),
    (-2010189.025055061, 111361.743448386),
    (-2010265.307499984, 110573.110516984),
    (-2009305.222904636, 110590.850726074),
    (-2009115.412615904, 110731.786831630),
    (-2009053.599104694, 110812.872130757),
    (-2009025.671474249, 111123.325789847),
    (-2009300.548510684, 111150.921670655),
    (-2009655.723586217, 111336.745881031)
]




# Function to calculate pixel or percent changes for a given dataset
def calculate_pixel_changes(estim_pixels, actual_pixels, mode="pixel"):
    changes = []

    for (est_x, est_y), (act_x, act_y) in zip(estim_pixels, actual_pixels):
        if mode == "pixel":
            # Calculate the radial pixel changes
            changes.append(np.sqrt((est_x - act_x) ** 2 + (est_y - act_y) ** 2))
        elif mode == "percent":
            # Calculate the percentage change relative to actual values
            radial_change = np.sqrt((est_x - act_x) ** 2 + (est_y - act_y) ** 2)
            actual_distance = np.sqrt(act_x ** 2 + act_y ** 2)
            percent_change = (radial_change / actual_distance) * 100 if actual_distance != 0 else 0
            changes.append(percent_change)
    return changes

# Collect changes from all datasets
changes_all = []
dataset_labels = []

datasets = [
    (estim_pixels_DATSETROT, actual_pixels_DATSETROT, "DATSETROT"),
    (estim_pixels_DATSETCPT, actual_pixels_DATSETCPT, "DATSETCPT"),
    (estim_pixels_DATSETROCK, actual_pixels_DATSETROCK, "DATSETROCK"),
    (estim_pixels_DATSETSAND, actual_pixels_DATSETSAND, "DATSETSAND"),
    (estim_pixels_DATSETAMAZ, actual_pixels_DATSETAMAZ, "DATSETAMAZ"),
]

# Choice of mode: "pixel" or "percent"
mode = "pixel"  # Switch to "pixel" or "percent" as needed

# Loop through datasets and accumulate changes
for estim, actual, label in datasets:
    changes = calculate_pixel_changes(estim, actual, mode=mode)
    dataset_labels.extend([label] * len(changes))  # Add labels for each dataset
    changes_all.extend(changes)

# Prepare the DataFrame for plotting
data = pd.DataFrame({
    'Dataset': dataset_labels,
    'Changes': changes_all
})

# Generate violin plot for radial changes
plt.figure(figsize=(10, 6))
sns.violinplot(x='Dataset', y='Changes', data=data)
plt.title(f'Violin Plot of Radial {mode.capitalize()} Changes by Dataset')
plt.xlabel('Dataset')
plt.ylabel(f'Radial {mode.capitalize()} Changes')

plt.tight_layout()
plt.show()

