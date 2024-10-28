# # # # # # # # import matplotlib.pyplot as plt

# # # # # # # # # Data from RMSE Comparison Table
# # # # # # # # datasets = ['CITY1', 'CITY2', 'ROCKY', 'DESERT', 'AMAZON']
# # # # # # # # lmeds_rmse = [51.78, 6.16, 19.04, 36.48, 31.57]  # RMSE values for LMEDS
# # # # # # # # ransac_rmse = [59.57, 5.84, 18.59, 35.90, 34.48]  # RMSE values for RANSAC

# # # # # # # # # Create RMSE plot
# # # # # # # # plt.figure(figsize=(10, 6))
# # # # # # # # plt.plot(datasets, lmeds_rmse, marker='o', label='LMEDS', color='blue')
# # # # # # # # plt.plot(datasets, ransac_rmse, marker='o', label='RANSAC', color='orange')

# # # # # # # # # Title and labels
# # # # # # # # plt.title('Outlier Technique RMSE Comparison Across Datasets')
# # # # # # # # plt.xlabel('Dataset')
# # # # # # # # plt.ylabel('RMSE GPS Error (m)')
# # # # # # # # plt.xticks(rotation=45)  # Rotate x-axis labels if needed
# # # # # # # # plt.grid()
# # # # # # # # plt.legend()
# # # # # # # # plt.tight_layout()

# # # # # # # # # # Show the plot
# # # # # # # # # plt.show()
# # # # # # # # import matplotlib.pyplot as plt

# # # # # # # # # Data from Runtime Comparison Table
# # # # # # # # datasets = ['CITY1', 'CITY2', 'ROCKY', 'DESERT', 'AMAZON']
# # # # # # # # lmeds_runtime = [68.72, 52.20, 69.44, 56.40, 50.34]  # Runtime values for LMEDS
# # # # # # # # ransac_runtime = [126.14, 64.65, 126.97, 126.97, 64.65]  # Runtime values for RANSAC

# # # # # # # # # Create Runtime plot
# # # # # # # # plt.figure(figsize=(10, 6))
# # # # # # # # plt.plot(datasets, lmeds_runtime, marker='o', label='LMEDS', color='blue')
# # # # # # # # plt.plot(datasets, ransac_runtime, marker='o', label='RANSAC', color='orange')

# # # # # # # # # Title and labels
# # # # # # # # plt.title('Runtime Comparison Across Datasets')
# # # # # # # # plt.xlabel('Dataset')
# # # # # # # # plt.ylabel('Runtime (seconds)')
# # # # # # # # plt.xticks(rotation=45)  # Rotate x-axis labels if needed
# # # # # # # # plt.grid()
# # # # # # # # plt.legend()
# # # # # # # # plt.tight_layout()

# # # # # # # # # Show the plot
# # # # # # # # plt.show()
# # # # # # # import matplotlib.pyplot as plt

# # # # # # # # Data for RMSE Comparison
# # # # # # # datasets = ['CITY1', 'CITY2', 'ROCKY', 'DESERT', 'AMAZON']
# # # # # # # rigid_rmse = [64.31, 7.42, 21.82, 29.01, 38.52]  # RMSE values for Rigid Transform
# # # # # # # affine_rmse = [127.77, 88.41, 57.01, 70.96, 118.77]  # RMSE values for Affine Transform
# # # # # # # homography_rmse = [168.35, 92.54, 58.69, 120.15, 242.97]  # RMSE values for Homography
# # # # # # # partial_affine_rmse = [64.11, 6.34, 19.07, 29.94, 40.08]  # RMSE values for Partial Affine 2D

# # # # # # # # Create RMSE Plot
# # # # # # # plt.figure(figsize=(10, 6))
# # # # # # # plt.plot(datasets, rigid_rmse, marker='o', label='Rigid Transform', color='blue')
# # # # # # # plt.plot(datasets, affine_rmse, marker='o', label='Affine Transform', color='orange')
# # # # # # # plt.plot(datasets, homography_rmse, marker='o', label='Homography', color='green')
# # # # # # # plt.plot(datasets, partial_affine_rmse, marker='o', label='Partial Affine 2D', color='red')

# # # # # # # # Title and labels for RMSE
# # # # # # # plt.title('Radial GPS RMSE Comparison Across Datasets for Translational Methods')
# # # # # # # plt.xlabel('Dataset')
# # # # # # # plt.ylabel('RMSE GPS Error (m)')
# # # # # # # plt.xticks(rotation=45)
# # # # # # # plt.grid()
# # # # # # # plt.legend()


# # # # # # # # Data for Runtime Comparison
# # # # # # # rigid_runtime = [89.05, 82.74, 67.46, 66.84, 77.43]  # Runtime values for Rigid Transform
# # # # # # # affine_runtime = [127.77, 88.41, 57.01, 70.96, 118.77]  # Runtime values for Affine Transform
# # # # # # # homography_runtime = [168.35, 92.54, 58.69, 120.15, 242.97]  # Runtime values for Homography
# # # # # # # partial_affine_runtime = [94.62, 82.77, 85.94, 72.43, 78.33]  # Runtime values for Partial Affine 2D

# # # # # # # # Create Runtime Plot
# # # # # # # plt.figure(figsize=(10, 6))
# # # # # # # plt.plot(datasets, rigid_runtime, marker='o', label='Rigid Transform', color='blue')
# # # # # # # plt.plot(datasets, affine_runtime, marker='o', label='Affine Transform', color='orange')
# # # # # # # plt.plot(datasets, homography_runtime, marker='o', label='Homography', color='green')
# # # # # # # plt.plot(datasets, partial_affine_runtime, marker='o', label='Partial Affine 2D', color='red')

# # # # # # # # Title and labels for Runtime
# # # # # # # plt.title('Runtime Comparison Across Datasets for Translational Methods')
# # # # # # # plt.xlabel('Dataset')
# # # # # # # plt.ylabel('Runtime (seconds)')
# # # # # # # plt.xticks(rotation=45)
# # # # # # # plt.grid()
# # # # # # # plt.legend()

# # # # # # # # Save Runtime Plot
# # # # # # # plt.tight_layout()
# # # # # # # plt.savefig('./Chapter 4/testresults/runtime_comparison_trans_methods.png')
# # # # # # # plt.close()
# # # # # # import matplotlib.pyplot as plt

# # # # # # # Data for RMSE Comparison
# # # # # # datasets = ['CITY1', 'CITY2', 'ROCKY', 'DESERT', 'AMAZON']
# # # # # # rigid_rmse = [64.31, 7.42, 21.82, 29.01, 38.52]  # RMSE values for Rigid Transform
# # # # # # affine_rmse = [127.77, 88.41, 57.01, 70.96, 118.77]  # RMSE values for Affine Transform
# # # # # # homography_rmse = [168.35, 92.54, 58.69, 120.15, 242.97]  # RMSE values for Homography
# # # # # # partial_affine_rmse = [64.11, 6.34, 19.07, 29.94, 40.08]  # RMSE values for Partial Affine 2D

# # # # # # # Create RMSE Plot
# # # # # # plt.figure(figsize=(10, 6))
# # # # # # plt.plot(datasets, rigid_rmse, marker='o', label='Rigid Transform', color='blue')
# # # # # # plt.plot(datasets, affine_rmse, marker='o', label='Affine Transform', color='orange')
# # # # # # plt.plot(datasets, homography_rmse, marker='o', label='Homography', color='green')
# # # # # # plt.plot(datasets, partial_affine_rmse, marker='o', label='Partial Affine 2D', color='red')

# # # # # # # Title and labels for RMSE
# # # # # # plt.title('Radial GPS RMSE Comparison Across Datasets for Translational Methods')
# # # # # # plt.xlabel('Dataset')
# # # # # # plt.ylabel('RMSE GPS Error (m)')
# # # # # # plt.xticks(rotation=45)
# # # # # # plt.grid()
# # # # # # plt.legend()

# # # # # # # Save RMSE Plot
# # # # # # plt.tight_layout()
# # # # # # plt.savefig('./REPORT/stellenbosch_ee_report_template-master/Chapter 4/testresults/rmse_comparison_trans_methods.png')
# # # # # # plt.close()

# # # # # # # Data for Runtime Comparison
# # # # # # rigid_runtime = [89.05, 82.74, 67.46, 66.84, 77.43]  # Runtime values for Rigid Transform
# # # # # # affine_runtime = [127.77, 88.41, 57.01, 70.96, 118.77]  # Runtime values for Affine Transform
# # # # # # homography_runtime = [168.35, 92.54, 58.69, 120.15, 242.97]  # Runtime values for Homography
# # # # # # partial_affine_runtime = [94.62, 82.77, 85.94, 72.43, 78.33]  # Runtime values for Partial Affine 2D

# # # # # # # Create Runtime Plot
# # # # # # plt.figure(figsize=(10, 6))
# # # # # # plt.plot(datasets, rigid_runtime, marker='o', label='Rigid Transform', color='blue')
# # # # # # plt.plot(datasets, affine_runtime, marker='o', label='Affine Transform', color='orange')
# # # # # # plt.plot(datasets, homography_runtime, marker='o', label='Homography', color='green')
# # # # # # plt.plot(datasets, partial_affine_runtime, marker='o', label='Partial Affine 2D', color='red')

# # # # # # # Title and labels for Runtime
# # # # # # plt.title('Runtime Comparison Across Datasets for Translational Methods')
# # # # # # plt.xlabel('Dataset')
# # # # # # plt.ylabel('Runtime (seconds)')
# # # # # # plt.xticks(rotation=45)
# # # # # # plt.grid()
# # # # # # plt.legend()

# # # # # # # Save Runtime Plot
# # # # # # plt.tight_layout()
# # # # # # plt.savefig('./REPORT/stellenbosch_ee_report_template-master/Chapter 4/testresults/runtime_comparison_trans_methods.png')
# # # # # # plt.close()
# # # # # import matplotlib.pyplot as plt

# # # # # # Data for RMSE and Percentage Change for Cross Correlation, Histogram, and SSIM
# # # # # datasets = ['CITY1', 'CITY2', 'ROCKY', 'DESERT', 'AMAZON']

# # # # # # RMSE values for each method
# # # # # cross_corr_rmse = [54.81, 5.34, 17.55, 32.51, 31.52]
# # # # # histogram_rmse = [56.17, 4.45, 16.83, 37.20, 36.35]
# # # # # ssim_rmse = [55.93, 5.53, 16.25, 41.01, 31.67]

# # # # # # Percentage change for each method
# # # # # cross_corr_pct_change = [5.89, 4.29, 11.54, 21.92, 0.06]
# # # # # histogram_pct_change = [20.93, -13.09, 6.73, 36.73, 15.38]
# # # # # ssim_pct_change = [11.79, -10.23, 3.04, 50.43, 0.73]

# # # # # # Adjusting the path to save the plots to the specified directory

# # # # # # Create RMSE Comparison Plot
# # # # # plt.figure(figsize=(10, 6))
# # # # # plt.plot(datasets, cross_corr_rmse, marker='o', label='Cross Correlation', linestyle='-', color='blue')
# # # # # plt.plot(datasets, histogram_rmse, marker='o', label='Histogram', linestyle='-', color='orange')
# # # # # plt.plot(datasets, ssim_rmse, marker='o', label='SSIM', linestyle='-', color='green')

# # # # # # Title and labels for RMSE Comparison
# # # # # plt.title('RMSE GPS Error Comparison Across Datasets for Various Methods')
# # # # # plt.xlabel('Dataset')
# # # # # plt.ylabel('RMSE GPS Error (m)')
# # # # # plt.xticks(rotation=45)
# # # # # plt.grid()
# # # # # plt.legend()

# # # # # # Save RMSE Comparison Plot to specified path
# # # # # plt.tight_layout()
# # # # # plt.savefig('./REPORT/stellenbosch_ee_report_template-master/Chapter 4/testresults/rmse_comparison_methods.png')
# # # # # plt.close()

# # # # # # Create Percentage Change Plot
# # # # # plt.figure(figsize=(10, 6))
# # # # # plt.plot(datasets, cross_corr_pct_change, marker='o', label='Cross Correlation', linestyle='-', color='blue')
# # # # # plt.plot(datasets, histogram_pct_change, marker='o', label='Histogram', linestyle='-', color='orange')
# # # # # plt.plot(datasets, ssim_pct_change, marker='o', label='SSIM', linestyle='-', color='green')

# # # # # # Title and labels for Percentage Change Comparison
# # # # # plt.title('Percentage Change in GPS Error with 10-degree Rotational Offset')
# # # # # plt.xlabel('Dataset')
# # # # # plt.ylabel('Percentage Change (%)')
# # # # # plt.xticks(rotation=45)
# # # # # plt.grid()
# # # # # plt.legend()

# # # # # # Save Percentage Change Plot to specified path
# # # # # plt.tight_layout()
# # # # # plt.savefig('./REPORT/stellenbosch_ee_report_template-master/Chapter 4/testresults/percentage_change_comparison_methods.png')
# # # # # plt.close()



# # # # import matplotlib.pyplot as plt

# # # # # Datasets
# # # # datasets = ['CITY1', 'CITY2', 'ROCKY', 'DESERT', 'AMAZON']

# # # # # RMSE values
# # # # local_retrofit_rmse = [46.56, 7.37, 17.32, 128.43, 34.54]
# # # # cross_corr_rmse = [49.88, 5.12, 15.77, 26.84, 31.50]
# # # # histogram_rmse = [46.35, 5.12, 15.77, 27.28, 31.50]
# # # # ssim_rmse = [50.03, 6.16, 15.77, 27.28, 31.44]

# # # # # Runtime values
# # # # local_retrofit_runtime = [84.56, 76.34, 70.22, 63.47, 90.69]
# # # # cross_corr_runtime = [59.62, 56.67, 54.14, 59.90, 65.92]
# # # # histogram_runtime = [57.06, 56.33, 56.68, 51.98, 59.09]
# # # # ssim_runtime = [91.60, 77.99, 83.31, 83.08, 114.07]

# # # # # RMSE Comparison Plot
# # # # plt.figure(figsize=(10, 6))
# # # # plt.plot(datasets, local_retrofit_rmse, marker='o', label='Local Retrofit', color='purple')
# # # # plt.plot(datasets, cross_corr_rmse, marker='o', label='Cross Correlation', color='blue')
# # # # plt.plot(datasets, histogram_rmse, marker='o', label='Histogram', color='orange')
# # # # plt.plot(datasets, ssim_rmse, marker='o', label='SSIM', color='green')
# # # # plt.title('RMSE Comparison Across Datasets for Global Matching Techniques')
# # # # plt.xlabel('Dataset')
# # # # plt.ylabel('RMSE (m)')
# # # # plt.xticks(rotation=45)
# # # # plt.legend()
# # # # plt.grid()
# # # # plt.tight_layout()
# # # # plt.savefig('./REPORT/stellenbosch_ee_report_template-master/Chapter 4/testresults/rmse_global_matching.png')
# # # # plt.close()

# # # # # Runtime Comparison Plot
# # # # plt.figure(figsize=(10, 6))
# # # # plt.plot(datasets, local_retrofit_runtime, marker='o', label='Local Retrofit', color='purple')
# # # # plt.plot(datasets, cross_corr_runtime, marker='o', label='Cross Correlation', color='blue')
# # # # plt.plot(datasets, histogram_runtime, marker='o', label='Histogram', color='orange')
# # # # plt.plot(datasets, ssim_runtime, marker='o', label='SSIM', color='green')
# # # # plt.title('Runtime Comparison Across Datasets for Global Matching Techniques')
# # # # plt.xlabel('Dataset')
# # # # plt.ylabel('Runtime (s)')
# # # # plt.xticks(rotation=45)
# # # # plt.legend()
# # # # plt.grid()
# # # # plt.tight_layout()
# # # # plt.savefig('./REPORT/stellenbosch_ee_report_template-master/Chapter 4/testresults/runtime_global_matching.png')
# # # # plt.close()


# # # import matplotlib.pyplot as plt

# # # # Datasets
# # # datasets = ['CITY1', 'CITY2', 'ROCKY', 'DESERT', 'AMAZON']

# # # # RMSE values
# # # partial_affine_2d_rmse = [70.18, 7.54, 27.98, 44.94, 46.98]
# # # affine_2d_rmse = [81.22, 7.80, 22.10, 43.11, 52.56]
# # # homography_rmse = [76.25, 7.16, 24.40, 48.56, 48.04]
# # # rigid_svd_rmse = [51.74, 6.16, 19.03, 36.48, 31.57]

# # # # Runtime values
# # # partial_affine_2d_runtime = [56.76, 52.20, 69.44, 56.40, 50.34]
# # # affine_2d_runtime = [65.27, 54.55, 76.16, 71.52, 58.16]
# # # homography_runtime = [68.79, 65.11, 78.92, 69.97, 58.27]
# # # rigid_svd_runtime = [83.33, 61.47, 77.94, 71.27, 74.77]

# # # # RMSE Comparison Plot
# # # plt.figure(figsize=(10, 6))
# # # plt.plot(datasets, partial_affine_2d_rmse, marker='o', label='Partial Affine 2D', color='purple')
# # # plt.plot(datasets, affine_2d_rmse, marker='o', label='Affine 2D', color='blue')
# # # plt.plot(datasets, homography_rmse, marker='o', label='Homography', color='orange')
# # # plt.plot(datasets, rigid_svd_rmse, marker='o', label='Rigid SVD', color='green')
# # # plt.title('RMSE Comparison Across Datasets for Rotational Estimators')
# # # plt.xlabel('Dataset')
# # # plt.ylabel('RMSE (m)')
# # # plt.xticks(rotation=45)
# # # plt.legend()
# # # plt.grid()
# # # plt.tight_layout()
# # # plt.savefig('./REPORT/stellenbosch_ee_report_template-master/Chapter 4/testresults/rmse_rotational_estimators.png')
# # # plt.close()

# # # # Runtime Comparison Plot
# # # plt.figure(figsize=(10, 6))
# # # plt.plot(datasets, partial_affine_2d_runtime, marker='o', label='Partial Affine 2D', color='purple')
# # # plt.plot(datasets, affine_2d_runtime, marker='o', label='Affine 2D', color='blue')
# # # plt.plot(datasets, homography_runtime, marker='o', label='Homography', color='orange')
# # # plt.plot(datasets, rigid_svd_runtime, marker='o', label='Rigid SVD', color='green')
# # # plt.title('Runtime Comparison Across Datasets for Rotational Estimators')
# # # plt.xlabel('Dataset')
# # # plt.ylabel('Runtime (s)')
# # # plt.xticks(rotation=45)
# # # plt.legend()
# # # plt.grid()
# # # plt.tight_layout()
# # # plt.savefig('./REPORT/stellenbosch_ee_report_template-master/Chapter 4/testresults/runtime_rotational_estimators.png')
# # # plt.close()

# # import matplotlib.pyplot as plt

# # # Datasets
# # datasets = ['CITY1', 'CITY2', 'ROCKY', 'DESERT', 'AMAZON']

# # # RMSE values
# # flann_rmse = [56.59, 4.70, 14.63, 71.20, 32.35]
# # bfmatcher_rmse = [53.89, 3.79, 16.36, 68.64, 33.24]

# # # Runtime values
# # flann_runtime = [42.72, 41.61, 41.96, 43.42, 53.62]
# # bfmatcher_runtime = [203.49, 228.59, 46.33, 52.59, 88.95]

# # # RMSE Comparison Plot
# # plt.figure(figsize=(10, 6))
# # plt.plot(datasets, flann_rmse, marker='o', label='FLANN', color='orange')
# # plt.plot(datasets, bfmatcher_rmse, marker='o', label='BFMatcher', color='blue')
# # plt.title('RMSE GPS Accuracy Comparison for BFMatcher and FLANN')
# # plt.xlabel('Dataset')
# # plt.ylabel('RMSE (m)')
# # plt.xticks(rotation=45)
# # plt.legend()
# # plt.grid()
# # plt.tight_layout()
# # plt.savefig('./REPORT/stellenbosch_ee_report_template-master/Chapter 4/testresults/rmse_flann_bf.png')
# # plt.close()

# # # Runtime Comparison Plot
# # plt.figure(figsize=(10, 6))
# # plt.plot(datasets, flann_runtime, marker='o', label='FLANN', color='orange')
# # plt.plot(datasets, bfmatcher_runtime, marker='o', label='BFMatcher', color='blue')
# # plt.title('Runtime Comparison for BFMatcher and FLANN')
# # plt.xlabel('Dataset')
# # plt.ylabel('Runtime (s)')
# # plt.xticks(rotation=45)
# # plt.legend()
# # plt.grid()
# # plt.tight_layout()
# # plt.savefig('./REPORT/stellenbosch_ee_report_template-master/Chapter 4/testresults/runtime_flann_bf.png')
# # plt.close()


# import matplotlib.pyplot as plt

# # Datasets
# datasets = ['CITY1', 'CITY2', 'ROCKY', 'DESERT', 'AMAZON']

# # RMSE values
# orb_rmse = [70.89, 10.56, 43.32, 80.66, 46.39]
# akaze_rmse = [66.80, 6.91, 22.48, 33.07, 39.27]
# superpoint_rmse = [72.66, 15.80, 114.93, 31.60, 329.19]

# # Runtime values
# orb_runtime = [75.25, 70.77, 111.64, 58.29, 75.34]
# akaze_runtime = [112.18, 104.16, 127.86, 108.86, 119.76]
# superpoint_runtime = [338.21, 292.11, 307.93, 277.73, 291.01]

# # RMSE Comparison Plot
# plt.figure(figsize=(10, 6))
# plt.plot(datasets, orb_rmse, marker='o', label='ORB', color='blue')
# plt.plot(datasets, akaze_rmse, marker='o', label='Dynamic AKAZE (3000 keypoints)', color='green')
# plt.plot(datasets, superpoint_rmse, marker='o', label='SuperPoint', color='orange')
# plt.title('RMSE for Various Local Detectors')
# plt.xlabel('Dataset')
# plt.ylabel('RMSE (m)')
# plt.xticks(rotation=45)
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.savefig('./REPORT/stellenbosch_ee_report_template-master/Chapter 4/testresults/rmse_detectors.png')
# plt.close()

# # Runtime Comparison Plot
# plt.figure(figsize=(10, 6))
# plt.plot(datasets, orb_runtime, marker='o', label='ORB', color='blue')
# plt.plot(datasets, akaze_runtime, marker='o', label='Dynamic AKAZE (3000 keypoints)', color='green')
# plt.plot(datasets, superpoint_runtime, marker='o', label='SuperPoint', color='orange')
# plt.title('Runtime for Various Local Detectors')
# plt.xlabel('Dataset')
# plt.ylabel('Runtime (s)')
# plt.xticks(rotation=45)
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.savefig('./REPORT/stellenbosch_ee_report_template-master/Chapter 4/testresults/runtime_detectors.png')
# plt.close()




# # Datasets
# datasets = ['CITY1', 'CITY2', 'ROCKY', 'DESERT', 'AMAZON']

# # Night and Evening RMSE GPS values
# night_rmse_gps = [68.65, 57.77, 91.42, 1588.15, 1271.68]
# evening_rmse_gps = [59.41, 6.44, 32.14, 156.34, 32.90]

# # Plotting RMSE for Night and Evening
# plt.figure(figsize=(10, 6))
# plt.plot(datasets, night_rmse_gps, marker='o', label='Night RMSE GPS (m)', color='purple')
# plt.plot(datasets, evening_rmse_gps, marker='o', label='Evening RMSE GPS (m)', color='orange')
# plt.title('Night and Evening RMSE GPS for Various Datasets')
# plt.xlabel('Dataset')
# plt.ylabel('RMSE GPS (m) (log scale)')
# plt.yscale('log') 
# plt.xticks(rotation=45)
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.savefig('./REPORT/stellenbosch_ee_report_template-master/Chapter 5/RESULTPLOTS/night_evening_rmse.png')
# plt.close()














import matplotlib.pyplot as plt




# Datasets
datasets = ['CITY1', 'CITY2', 'ROCKY', 'DESERT', 'AMAZON']

# RMSE values
partial_affine_2d_rmse = [70.18, 7.54, 27.98, 44.94, 46.98]
affine_2d_rmse = [81.22, 7.80, 22.10, 43.11, 52.56]
homography_rmse = [76.25, 7.16, 24.40, 48.56, 48.04]
rigid_svd_rmse = [51.74, 6.16, 19.03, 36.48, 31.57]

# Runtime values
partial_affine_2d_runtime = [56.76, 52.20, 69.44, 56.40, 50.34]
affine_2d_runtime = [65.27, 54.55, 76.16, 71.52, 58.16]
homography_runtime = [68.79, 65.11, 78.92, 69.97, 58.27]
rigid_svd_runtime = [83.33, 61.47, 77.94, 71.27, 74.77]

# RMSE Comparison Plot
plt.figure(figsize=(10, 6))
plt.plot(datasets, partial_affine_2d_rmse, marker='o', label='Partial Affine 2D', color='purple')
plt.plot(datasets, affine_2d_rmse, marker='o', label='Affine 2D', color='blue')
plt.plot(datasets, homography_rmse, marker='o', label='Homography', color='orange')
plt.plot(datasets, rigid_svd_rmse, marker='o', label='Rigid SVD', color='green')
plt.title('RMSE Comparison Across Datasets for Planar Transformation Methods')
plt.xlabel('Dataset')
plt.ylabel('RMSE (m)')
plt.xticks(rotation=45)
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('./REPORT/stellenbosch_ee_report_template-master/Chapter 4/testresults/rmse_planar_estimators.png')

# plt.close()


















# # Data for RMSE Comparison
# datasets = ['CITY1', 'CITY2', 'ROCKY', 'DESERT', 'AMAZON']
# rigid_rmse = [64.31, 7.42, 21.82, 29.01, 38.52]  # RMSE values for Rigid Transform
# affine_rmse = [127.77, 88.41, 57.01, 70.96, 118.77]  # RMSE values for Affine Transform
# homography_rmse = [168.35, 92.54, 58.69, 120.15, 242.97]  # RMSE values for Homography
# partial_affine_rmse = [64.11, 6.34, 19.07, 29.94, 40.08]  # RMSE values for Partial Affine 2D



# Data for Runtime Comparison
rigid_runtime = [89.05, 82.74, 55.46, 66.84, 77.43]  # Runtime values for Rigid Transform
affine_runtime = [127.77, 88.41, 57.01, 70.96, 118.77]  # Runtime values for Affine Transform
homography_runtime = [168.35, 92.54, 58.69, 120.15, 242.97]  # Runtime values for Homography
partial_affine_runtime = [94.62, 82.77, 85.94, 72.43, 78.33]  # Runtime values for Partial Affine 2D

# Create Runtime Plot
plt.figure(figsize=(10, 6))
plt.plot(datasets, rigid_runtime, marker='o', label='Rigid Transform', color='blue')
plt.plot(datasets, affine_runtime, marker='o', label='Affine Transform', color='orange')
plt.plot(datasets, homography_runtime, marker='o', label='Homography', color='green')
plt.plot(datasets, partial_affine_runtime, marker='o', label='Partial Affine 2D', color='red')

# Title and labels for Runtime
plt.title('Runtime Comparison Across Datasets for Planar Transformation Methods')
plt.xlabel('Dataset')
plt.ylabel('Runtime (seconds)')
plt.xticks(rotation=45)
plt.grid()
plt.legend()

# Save Runtime Plot
plt.tight_layout()
plt.savefig('./REPORT/stellenbosch_ee_report_template-master/Chapter 4/testresults/runtime_planar_estimators.png')
# plt.close()
plt.show()