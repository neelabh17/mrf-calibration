for i, label in enumerate(uncal_labels):
                ax = plt.subplot(rows, cols, i + 2)
                ax.set_title("Uncalibrated-" + classes[label])
                ax.imshow(uncal_all_class_prob[label], cmap="nipy_spectral")

                ax.axis("off")

            for i, label in enumerate(cal_labels):
                ax = plt.subplot(rows, cols, cols + i + 2)
                ax.set_title("Calibrated-" + classes[label])
                ax.imshow(cal_all_class_prob[label], cmap="nipy_spectral")
                ax.axis("off")

            for i, label in enumerate(cal_labels):
                ax = plt.subplot(rows, cols, 2 * cols + i + 2)
                
                min_dif=np.min(uncal_all_class_prob[label] - cal_all_class_prob[label])
                max_dif=np.max(uncal_all_class_prob[label] - cal_all_class_prob[label])

                dif_map=np.where((uncal_all_class_prob[label] - cal_all_class_prob[label])>0,(uncal_all_class_prob[label] - cal_all_class_prob[label]),0)
                
                ax.set_title(
                    "decrease: "
                    + classes[label]
                    + " max={:0.3f}".format(
                        max_dif
                    )
                )
                ax.imshow(
                    dif_map
                    / max_dif,
                    cmap="nipy_spectral",
                )
                ax.axis("off")
            
            for i, label in enumerate(cal_labels):
                ax = plt.subplot(rows, cols, 3 * cols + i + 2)
                
                min_dif=np.min(uncal_all_class_prob[label] - cal_all_class_prob[label])
                max_dif=np.max(uncal_all_class_prob[label] - cal_all_class_prob[label])

                dif_map=np.where((uncal_all_class_prob[label] - cal_all_class_prob[label])<0,(uncal_all_class_prob[label] - cal_all_class_prob[label]),0)
                
                ax.set_title(
                    "increase: "
                    + classes[label]
                    + " max={:0.3f}".format(
                        -min_dif
                    )
                )
                ax.imshow(
                    dif_map
                    / min_dif,
                    cmap="nipy_spectral",
                )
                ax.axis("off")
            
            acc_cal_mask=(cal_labelmap==gt_label).astype(np.float32)
            acc_uncal_mask=(uncal_labelmap==gt_label).astype(np.float32)
            print(np.sum(acc_cal_mask)-np.sum(acc_uncal_mask))

            for i, label in enumerate(cal_labels):
                ax = plt.subplot(rows, cols, 4 * cols + i + 2)
                dif_map=np.where((acc_uncal_mask - acc_cal_mask)<0,(acc_uncal_mask - acc_cal_mask),0)
                
                ax.set_title(
                    "increase: "
                    + classes[label]
                    + " max={:0.3f}".format(
                        -1
                    )
                )
                ax.imshow(
                    dif_map/-1,
                    cmap="nipy_spectral",
                )

            for i, label in enumerate(cal_labels):
                ax = plt.subplot(rows, cols, 5 * cols + i + 2)
                dif_map=np.where((acc_uncal_mask - acc_cal_mask)>0,(acc_uncal_mask - acc_cal_mask),0)
                
                ax.set_title(
                    "increase: "
                    + classes[label]
                    + " max={:0.3f}".format(
                        1
                    )
                )
                ax.imshow(
                    dif_map/1,
                    cmap="nipy_spectral",
                )