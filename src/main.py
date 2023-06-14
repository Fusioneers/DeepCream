import os
import cv2
from cloud_analysis.analysis import Analysis
from cloud_detection.cloud_detection import CloudDetection
import numpy as np
from pareidolia.pareidolia import Pareidolia

INPUT_DIR = "data-test"

def main():

    pareidolia = Pareidolia()

    # Delete the results folder if it exists
    if os.path.exists("results"):
        os.system("rm -rf results")

    # List the contents for the data directory

    for folder in os.listdir(INPUT_DIR):
        print(folder)
        for file in os.listdir(f"{INPUT_DIR}/{folder}"):
            if file != "masked.png":
                continue

            # Open the image with OpenCV
            original = cv2.imread(f"{INPUT_DIR}/{folder}/{file}")

            # Get all non black pixels from the image
            mask = cv2.inRange(original, np.array(
                [5, 5, 5]), np.array([255, 255, 255]))

            # Create the results folder if it doesn't exist
            if not os.path.exists(f"results/{folder}"):
                os.makedirs(f"results/{folder}")

            # Copy the original image to results/{folder}/original.png
            cv2.imwrite(f"results/{folder}/masked.png", original)

            # Save the mask to results/{folder}/mask.png
            cv2.imwrite(f"results/{folder}/mask.png", mask)

            max_num_clouds = 5
            max_border_proportion = 1

            analysis = Analysis(original, mask, max_num_clouds,
                                max_border_proportion)

            analysis_results = analysis.evaluate()

            if analysis_results.empty:
                print("No clouds found")
                continue

            masks = [np.tile(cloud.mask[:, :, np.newaxis], 3)
                     for cloud in analysis.clouds]
            pareidolia_results = pareidolia.evaluate_clouds(masks)

            for i, mask in enumerate(masks):
                os.makedirs(f"results/{folder}/cloud_{i}")

                # Save the mask
                cv2.imwrite(f"results/{folder}/cloud_{i}/mask.png", mask)

                # Save the cloud cutout from the original image with the mask
                cv2.imwrite(f"results/{folder}/cloud_{i}/cloud.png", cv2.bitwise_and(
                    original, mask))

                # Save the results from analysis
                with open(f"results/{folder}/cloud_{i}/data.txt", "w") as f:
                    f.write(str(analysis_results.iloc[i]))

                # Save the results from pareidolia
                with open(f"results/{folder}/cloud_{i}/pareidolia.txt", "w") as f:
                    results = pareidolia_results.iloc[i]

                    # Get the top 3 results
                    results = results.sort_values(ascending=False)[:3]
                    top_3 = results.index.tolist()

                    # Write the top 3 results to the file
                    f.write("\n".join(f"{i}: {j}" for i,
                            j in zip(top_3, results)))


if __name__ == '__main__':
    main()
