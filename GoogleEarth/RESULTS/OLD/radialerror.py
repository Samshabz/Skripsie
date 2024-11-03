import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns




# For Dataset DATSETROT
estim_pixels_DATSETROT = [
    (424705.729276799, 903473.767292299),
    (424409.698721405, 903558.233969463),
    (424076.513694823, 903913.454724287),
    (423948.493328866, 904272.605740991),
    (424119.264133204, 904476.033116474),
    (424360.160396090, 904707.758923616),
    (424476.110481576, 905011.995590373),
    (424402.143670195, 905385.392438020),
    (424216.133800685, 905719.292526812),
    (424055.050771255, 905846.079422976),
    (423729.100589019, 905822.429690331),
    (423426.632799929, 905794.677602502),
    (423105.555655199, 905895.818131606),
    (422831.670289880, 906132.745309303)
]

actual_pixels_DATSETROT = [
    (424701.879011917, 903456.497300373),
    (424414.801071789, 903574.548075606),
    (424053.412483904, 903913.379705081),
    (423927.180120735, 904291.586263975),
    (424136.156482997, 904456.709323249),
    (424372.355981284, 904707.539465733),
    (424491.109530233, 904999.002759065),
    (424391.191168392, 905377.061291909),
    (424215.289281339, 905720.777781049),
    (424056.352244562, 905844.379533173),
    (423724.384021566, 905833.795670566),
    (423418.655360616, 905809.741437368),
    (423106.387742999, 905909.066917218),
    (422841.620090201, 906142.578011800)
]

# For Dataset DATSETCPT
estim_pixels_DATSETCPT = [
    (546792.006980915, 1182381.680883119),
    (546667.924404617, 1182724.437321964),
    (546058.384845164, 1182921.225348541),
    (545703.172889091, 1183122.167257675),
    (545451.237959307, 1183408.495398590),
    (545084.793630524, 1183630.156549520),
    (544789.088307159, 1183922.967914691),
    (545139.063627362, 1184295.602729352),
    (545494.821652657, 1184507.618917050),
    (545899.510350524, 1184519.054758719),
    (546216.357346457, 1184553.358430448),
    (545937.293250798, 1184974.767465310),
    (545674.151745477, 1185287.878695888),
    (545350.682054972, 1185535.621026711)
]

actual_pixels_DATSETCPT = [
    (546796.098178374, 1182380.870152775),
    (546664.761939655, 1182725.503514955),
    (546059.576512630, 1182923.544165024),
    (545702.554649825, 1183121.584815094),
    (545449.174767906, 1183408.181802816),
    (545081.994411632, 1183632.091755023),
    (544788.727495207, 1183923.630070120),
    (545135.349414032, 1184294.714067070),
    (545494.974197846, 1184508.935142071),
    (545899.633648772, 1184519.399129453),
    (546215.261191711, 1184556.023085288),
    (545937.286989924, 1184975.163913174),
    (545675.073166376, 1185288.405313210),
    (545349.880850827, 1185535.471681936)
]

# For Dataset DATSETROCK
estim_pixels_DATSETROCK = [
    (645408.512966997, 1143069.467125723),
    (645799.249326898, 1142859.652672276),
    (645352.038988787, 1143141.674492344),
    (645226.736975717, 1143869.536145306),
    (644911.983146617, 1143786.999773447),
    (645243.246939530, 1143448.005783822),
    (644657.047553829, 1143614.726582623),
    (644111.402458115, 1143230.675042805),
    (643801.739118396, 1143380.683833783),
    (643770.735502226, 1143385.933979210),
    (642917.440159014, 1143309.447979634),
    (642177.376540807, 1143630.680300689),
    (641803.448642168, 1144189.246345547),
    (641732.781569198, 1144352.848291490)
]

actual_pixels_DATSETROCK = [
    (645407.675689632, 1143069.795158848),
    (645791.684092618, 1142860.997110658),
    (645352.742945273, 1143141.641812539),
    (645227.226689103, 1143873.290298511),
    (644906.003968123, 1143801.041142559),
    (645239.569621061, 1143455.794827696),
    (644662.174159911, 1143601.802523081),
    (644110.655451668, 1143236.632346270),
    (643801.328900930, 1143381.432534870),
    (643770.514895223, 1143385.256306355),
    (642925.445654017, 1143300.127078033),
    (642170.269587924, 1143621.927636160),
    (641810.060838531, 1144186.537683580),
    (641726.322869750, 1144354.884254482)
]

# For Dataset DATSETSAND
estim_pixels_DATSETSAND = [
    (-160186.489888970, -706995.691624257),
    (-160121.989993245, -706762.444860379),
    (-160229.906178928, -706564.470397061),
    (-160412.403254350, -706595.851084545),
    (-160635.235168030, -706378.294701656),
    (-160771.540097292, -706086.825046535),
    (-160642.304977789, -705729.707777519),
    (-160332.917627746, -705569.165145745),
    (-160002.349882395, -705437.697709180),
    (-159909.537671912, -705233.816696991),
    (-160127.971782797, -705057.321085163),
    (-160026.892881272, -704753.322873397),
    (-159732.171442254, -704488.664727085),
    (-159846.578783319, -704126.554169941)
]

actual_pixels_DATSETSAND = [
    (-160198.633361978, -706994.658247095),
    (-160111.295823539, -706761.389162332),
    (-160227.555647411, -706566.465680543),
    (-160417.955958122, -706594.126441022),
    (-160624.100201263, -706369.045740228),
    (-160762.096105955, -706086.546493312),
    (-160656.758679849, -705719.467231502),
    (-160324.109688486, -705577.069237121),
    (-160008.492103094, -705433.872376012),
    (-159915.845281699, -705238.649319201),
    (-160121.982221411, -705053.212379815),
    (-160034.230580316, -704744.550247537),
    (-159726.679511842, -704498.499295115),
    (-159824.034680004, -704113.145956887)
]

# For Dataset DATSETAMAZ
estim_pixels_DATSETAMAZ = [
    (-2078524.714389769, 110028.038695669),
    (-2078867.746042622, 109922.551686032),
    (-2078948.437655320, 109678.012812354),
    (-2078918.689001731, 109508.139227323),
    (-2078965.246917296, 109249.317659918),
    (-2079011.316338422, 108891.761120940),
    (-2078667.644930189, 108900.982064732),
    (-2078434.770875788, 108845.661635054),
    (-2078028.088097279, 108925.992146617),
    (-2077842.365015154, 109061.655686421),
    (-2077758.848097560, 109135.730570247),
    (-2077757.371770059, 109436.705877882),
    (-2078011.805105307, 109488.911287298),
    (-2078383.294975149, 109656.868254995)
]

actual_pixels_DATSETAMAZ = [
    (-2078530.371972869, 110022.251942257),
    (-2078884.700532446, 109943.274079266),
    (-2078943.057326625, 109679.338003059),
    (-2078925.388759385, 109506.733868970),
    (-2078976.304004506, 109248.621858637),
    (-2079021.948842444, 108902.619399658),
    (-2078652.062408275, 108886.294366525),
    (-2078420.656814352, 108842.702115891),
    (-2078029.026694792, 108920.091597280),
    (-2077832.724368467, 109058.898500615),
    (-2077768.796663970, 109138.758797829),
    (-2077739.913831212, 109444.522256223),
    (-2078024.192423620, 109471.701230303),
    (-2078391.515470463, 109654.718088228)
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


dataset_names = ['DATSETROT', 'DATSETCPT', 'DATSETROCK', 'DATSETSAND', 'DATSETAMAZ']
colors = ['blue', 'green', 'red', 'purple', 'orange']  # Different colors for each dataset


# Step 6: Plot the distribution of radial normalized deviations with frequency and normal curve
plt.figure(figsize=(10, 6))

# Create histograms for each dataset
for i, (estim, actual) in enumerate(datasets):
    x_changes, y_changes = calculate_pixel_changes(estim, actual)
    radial_deviations_signed = [np.sqrt(x**2 + y**2) * np.sign(x) for x, y in zip(x_changes, y_changes)]
    radial_bins_signed = np.round(radial_deviations_signed).astype(int)

    # Create a histogram for the current dataset
    sns.histplot(radial_bins_signed, bins=50, color=colors[i], stat='count', label=dataset_names[i], alpha=0.6)

# Calculate the normal curve for the entire distribution
normal_curve_signed = (1/(std_dev_signed * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_values_signed - mean_signed) / std_dev_signed) ** 2)

# Plot the normal curve on the histogram
plt.plot(x_values_signed, normal_curve_signed * max(np.histogram(radial_bins_signed, bins=50)[0]) / max(normal_curve_signed), color='black', label='Normal Curve')

# Add a vertical line at zero
plt.axvline(0, color='red', linestyle='--', label='Zero Center')
plt.title('Distribution of Radial Normalized Deviations (Centered Around Zero)')
plt.xlabel('Radial Deviations')
plt.ylabel('Frequency')
plt.legend(title='Datasets')
plt.grid()
plt.show()
