import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns




# For Dataset DATSETROT
estim_pixels_DATSETROT = [
    (424306.842848935, 903641.766609983),
    (424010.758676450, 903724.292558043),
    (423636.634759266, 904070.482934271),
    (423550.666837887, 904441.639543082),
    (423719.423298785, 904643.475218895),
    (423958.787272845, 904893.284555666),
    (424078.302050807, 905180.668040199),
    (424002.020278437, 905554.143686031),
    (423817.179276469, 905887.538890792),
    (423655.672744325, 906014.160968569),
    (423329.793052396, 905990.706720479),
    (423028.107801047, 905963.675707130),
    (422707.535592065, 906063.183302816),
    (422432.996443040, 906301.654126529)
]

actual_pixels_DATSETROT = [
    (424302.352597561, 903624.250924117),
    (424015.544717977, 903742.323618994),
    (423654.496096298, 904081.218162659),
    (423528.382482694, 904459.494946872),
    (423737.162256288, 904624.648666164),
    (423973.139556504, 904875.525382739),
    (424091.781391372, 905167.042794917),
    (423991.957024936, 905545.171525595),
    (423816.220612664, 905888.951835945),
    (423657.433091459, 906012.576538417),
    (423325.777158284, 906001.990710601),
    (423020.336103020, 905977.932011018),
    (422708.362242431, 906077.275933604),
    (422443.843662398, 906310.830386479)
]

# For Dataset DATSETCPT
estim_pixels_DATSETCPT = [
    (547022.578125530, 1183863.612601986),
    (546897.318106240, 1184207.652011815),
    (546287.212638249, 1184404.702311126),
    (545933.751176742, 1184605.664170423),
    (545681.474976044, 1184891.754404632),
    (545314.565961376, 1185114.597423722),
    (545019.072106023, 1185407.056180076),
    (545369.956980299, 1185780.178620864),
    (545725.772106468, 1185993.055967430),
    (546130.036305914, 1186004.234025993),
    (546446.491226195, 1186038.598310570),
    (546166.463390818, 1186460.687856771),
    (545906.361081027, 1186772.161369286)
]

actual_pixels_DATSETCPT = [
    (547026.583065096, 1183863.497501317),
    (546895.191465671, 1184208.563010922),
    (546289.750941553, 1184406.851990858),
    (545932.578587288, 1184605.140970794),
    (545679.091900961, 1184892.097332169),
    (545311.756771229, 1185116.288052635),
    (545018.366237259, 1185408.191937482),
    (545365.134263771, 1185779.741249202),
    (545724.910636226, 1185994.230943252),
    (546129.740658756, 1186004.708051781),
    (546445.501244668, 1186041.377931632),
    (546167.409871503, 1186461.044334373),
    (545905.085519874, 1186774.678518392)
]

# For Dataset DATSETROCK
estim_pixels_DATSETROCK = [
    (643963.635077606, 1139981.721811950),
    (644354.028360593, 1139772.566838494),
    (643907.023175435, 1140054.222211114),
    (643777.215297394, 1140777.356228280),
    (643470.094612187, 1140697.995800588),
    (643797.901812573, 1140360.865763810),
    (643212.678871098, 1140524.131916451),
    (642669.619094224, 1140142.263574832),
    (642359.898861204, 1140292.240343540),
    (642328.278710896, 1140297.678920978),
    (641475.826178923, 1140221.418333976),
    (640735.742075722, 1140539.935731494),
    (640365.780439391, 1141098.068311811),
    (640294.553931371, 1141262.334260680)
]

actual_pixels_DATSETROCK = [
    (643962.177742255, 1139982.196980706),
    (644345.326094487, 1139773.962926487),
    (643907.368028922, 1140053.849566110),
    (643782.132887307, 1140783.521762846),
    (643461.629598092, 1140711.467762398),
    (643794.448175213, 1140367.154008169),
    (643218.345887333, 1140512.767315481),
    (642668.062397062, 1140148.583516560),
    (642359.428634636, 1140292.992578739),
    (642328.683642025, 1140296.806021659),
    (641485.507074085, 1140211.906739794),
    (640732.022350526, 1140532.838067695),
    (640372.620348601, 1141095.923021056),
    (640289.069924942, 1141263.814863325)
]

# For Dataset DATSETSAND
estim_pixels_DATSETSAND = [
    (-159586.861709297, -706192.598655812),
    (-159512.904949154, -705959.783388675),
    (-159630.993921165, -705761.832208575),
    (-159811.920884809, -705792.596760078),
    (-160034.427416946, -705575.032797845),
    (-160169.881246543, -705284.553572104),
    (-160040.621053750, -704926.409815583),
    (-159732.053523411, -704766.773221358),
    (-159403.527921105, -704636.302735075),
    (-159310.126858615, -704432.694807589),
    (-159529.077215461, -704255.976275947),
    (-159428.791525029, -703951.999124976),
    (-159134.029311420, -703685.898296420),
    (-159243.028035472, -703325.724556677)
]

actual_pixels_DATSETSAND = [
    (-159598.859836861, -706191.146748891),
    (-159511.849284628, -705958.142778425),
    (-159627.673839091, -705763.440830501),
    (-159817.361303113, -705791.070154054),
    (-160022.733755278, -705566.245261318),
    (-160160.213012053, -705284.067079619),
    (-160055.269962592, -704917.405009656),
    (-159723.866387959, -704775.168853027),
    (-159409.430454793, -704632.134737595),
    (-159317.130497223, -704437.133555120),
    (-159522.495673303, -704251.907367982),
    (-159435.072568789, -703943.596035486),
    (-159128.672952013, -703697.824724172),
    (-159225.663628599, -703312.909346591)
]

estim_pixels_DATSETAMAZ = [
    (-2015214.650763362, 110931.518558107),
    (-2015554.234887714, 110827.586271302),
    (-2015623.388786934, 110578.997499846),
    (-2015592.021905049, 110410.888819745),
    (-2015638.488339936, 110151.064772837),
    (-2015680.626678861, 109790.454554812),
    (-2015338.661309562, 109796.890448298),
    (-2015117.588249616, 109761.572912502),
    (-2014735.797047199, 109823.328068487),
    (-2014551.846688818, 109959.433835221),
    (-2014466.002264184, 110033.703333187),
    (-2014458.678633595, 110339.534503982),
    (-2014722.009614524, 110390.308060070),
    (-2015073.679566519, 110561.510196290)
]

actual_pixels_DATSETAMAZ = [
    (-2015216.720271542, 110927.703891302),
    (-2015560.255707616, 110848.076062884),
    (-2015616.834908458, 110581.967867579),
    (-2015599.704539517, 110407.943250388),
    (-2015649.068866673, 110147.707051369),
    (-2015693.323327348, 109798.857090222),
    (-2015334.703921642, 109782.397706695),
    (-2015110.347122085, 109738.446704193),
    (-2014730.646360555, 109816.473079079),
    (-2014540.323555764, 109956.422323885),
    (-2014478.343138865, 110036.939848710),
    (-2014450.340099656, 110345.219653702),
    (-2014725.959345043, 110372.622303034),
    (-2015082.093446195, 110557.145337826)
]











def calculate_pixel_changes(estim_pixels, actual_pixels):
    x_changes = []
    y_changes = []

    for (est_x, est_y), (act_x, act_y) in zip(estim_pixels, actual_pixels):
        x_changes.append(est_x - act_x)  # X coordinate change
        y_changes.append(est_y - act_y)  # Y coordinate change
    return x_changes, y_changes

# Example datasets (replace with your actual data variables)
datasets = [
    (estim_pixels_DATSETROT, actual_pixels_DATSETROT),
    (estim_pixels_DATSETCPT, actual_pixels_DATSETCPT),
    (estim_pixels_DATSETROCK, actual_pixels_DATSETROCK),
    (estim_pixels_DATSETSAND, actual_pixels_DATSETSAND),
    (estim_pixels_DATSETAMAZ, actual_pixels_DATSETAMAZ),
]

# Collect pixel changes from all datasets
x_changes_all = []
y_changes_all = []

# Loop through datasets and accumulate changes
for estim, actual in datasets:
    x_changes, y_changes = calculate_pixel_changes(estim, actual)
    x_changes_all.extend(x_changes)
    y_changes_all.extend(y_changes)

# Step 1: Calculate radial normalized deviations
# Calculate radial normalized deviations (without absolute values)
radial_deviations_signed = [np.sqrt(x**2 + y**2) * np.sign(x) for x, y in zip(x_changes_all, y_changes_all)]

# Step 2: Round off the deviations for creating bins
radial_bins_signed = np.round(radial_deviations_signed).astype(int)

# Step 3: Create a DataFrame for frequency counting of radial deviations
data_radial_signed = pd.DataFrame({'radial': radial_bins_signed})

# Step 4: Calculate mean and standard deviation for the radial deviations
mean_signed = np.mean(data_radial_signed['radial'])
std_dev_signed = np.std(data_radial_signed['radial'])

# Step 5: Create a range for the normal distribution curve
x_values_signed = np.linspace(min(data_radial_signed['radial']), max(data_radial_signed['radial']), 1000)
normal_curve_signed = (1/(std_dev_signed * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_values_signed - mean_signed) / std_dev_signed) ** 2)



radial_deviations_raw = radial_deviations_signed

mean_signed1 = np.mean(radial_deviations_raw)
std_dev_signed1 = np.std(radial_deviations_raw)
median_signed1 = np.median(radial_deviations_raw)
variance_signed1 = np.var(radial_deviations_raw)
min_signed1 = np.min(radial_deviations_raw)
max_signed1 = np.max(radial_deviations_raw)
q1_signed1 = np.percentile(radial_deviations_raw, 25)
q3_signed1 = np.percentile(radial_deviations_raw, 75)
iqr_signed1 = q3_signed1 - q1_signed1  # Interquartile range

# Collecting statistics in a dictionary for output
stats_summary = {
    'Mean': mean_signed,
    'Standard Deviation': std_dev_signed,
    'Median': median_signed1,
    'Variance': variance_signed1,
    'Minimum': min_signed1,
    'Maximum': max_signed1,
    'Q1': q1_signed1,
    'Q3': q3_signed1,
    'IQR': iqr_signed1
}

# Displaying the statistics
print(stats_summary)


# Step 6: Plot the distribution of radial normalized deviations with frequency and normal curve
plt.figure(figsize=(10, 6))
sns.histplot(data_radial_signed['radial'], bins=50, color='blue', stat='count', label='Frequency', alpha=0.6)
plt.plot(x_values_signed, normal_curve_signed * max(np.histogram(data_radial_signed['radial'], bins=50)[0]) / max(normal_curve_signed), color='orange', label='Normal Curve')
plt.axvline(0, color='red', linestyle='--', label='Zero Center')
plt.title('Distribution of Radial Normalized Deviations (Centered Around Zero)')
plt.xlabel('Radial Deviations')
plt.ylabel('Frequency')
plt.legend()
plt.grid()
plt.show()




