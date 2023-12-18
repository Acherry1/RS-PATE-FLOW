# # 示例使用
# num_classes = 5  # 例如，有5个类别
# class_losses = ClassLossMeter(num_classes)
#
# # 模拟训练循环
# for epoch in range(num_epochs):
#     for batch_data in data_loader:
#         # 模拟计算每个样本的损失和真实类别索引
#         # 假设 class_indices 是样本的真实类别索引列表， losses 是损失列表
#         class_indices = compute_true_class_indices(batch_data)
#         losses = compute_losses(batch_data)
#
#         # 更新 ClassLossMeter
#         class_losses.update(class_indices, losses, len(batch_data))
#
#     # 打印当前 epoch 的每个类别的平均损失
#     avg_losses = class_losses.average_losses()
#     print(f'Epoch: {epoch + 1}, Average Losses per Class: {avg_losses}')
#
#     # 重置 ClassLossMeter，准备下一个 epoch
#     class_losses.reset()
# --------------------------------------------------------------------------------------------------------

# list_images = os.listdir(data_path)
# len_data = len(list_images)
# print(len_data)

# # DATASETS AND DATALOADERS

# # datasets
# ds_train = datasets.ImageFolder(str(data_path), transform = train_transform, target_transform = lambda t:torch.tensor(t).long())
# ds_test = datasets.ImageFolder(str(data_path), transform = test_transform, target_transform = lambda t: torch.tensor(t).long())
# def pil_loader(path: str):
#     # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
#     with open(path, 'rb') as f:
#         img = Image.open(f)
#         return img.convert('RGB')

# TODO: specify the return type
# def accimage_loader(path: str):
#     import accimage
#     try:
#         return accimage.Image(path)
#     except IOError:
#         # Potentially a decoding problem, fall back to PIL.Image
#         return pil_loader(path)


# def default_loader(path: str):
#     from torchvision import get_image_backend
#     if get_image_backend() == 'accimage':
#         return accimage_loader(path)
#     else:
#         return pil_loader(path)


# def add_noise(predicted_labels, epsilon):
#     noisy_labels = []
#     # print(predicted_labels)
#     for predicts in predicted_labels:
#         # print(predicts)
#         label_counts = np.bincount(predicts, minlength=2)
#         # print(label_counts)
#         # 在标签上添加拉普拉斯噪声
#         epsilon = epsilon
#         beta = 1 / epsilon
#         for i in range(len(label_counts)):
#             label_counts[i] += np.random.laplace(0, beta, 1)
#         new_label = np.argmax(label_counts)
#         noisy_labels.append(new_label)
#     # print(len(noisy_labels))
#     # 返回添加噪声后的标签
#     return np.array(noisy_labels)


# def get_noise_labels(epsilon, result_true_xlsx):
#     # df_result_path = os.path.join(BASE_DIR, "result_label.xlsx")
#     #     df_result = pd.read_excel(df_result_path)
#     #     result_true_xlsx.to_excel("old_data.xlsx", index=False)
#     a_all = []
#     for i in result_true_xlsx.itertuples():
#         a = [1 for _ in range(getattr(i, "one_count"))]
#         a += [0 for _ in range(getattr(i, "zeo_count"))]
#         a_all.append(a)
#     labels_with_noise = add_noise(a_all, epsilon=epsilon)
#     # a_str = "labels_with_noise_{}".format(epsilon)
#     result_true_xlsx["labels_with_noise_{}".format(epsilon)] = labels_with_noise
#     agg_acc = len(result_true_xlsx[result_true_xlsx["labels_with_noise_{}".format(epsilon)] == result_true_xlsx["labels"]]) / len(result_true_xlsx)
#     print("epsilon:{},聚合准确率：{}".format(epsilon, agg_acc))
#     #     result_true_xlsx.to_excel("new_data.xlsx", index=False)
#     return result_true_xlsx

# 选择每个类别的前 5 个数据点
# selected_data = []
#
# for unique_label in range(10):
#     # 找到当前类别的索引
#     indices_for_label = np.where(labels == unique_label)[0]
#
#     # 选择前 5 个数据点
#     selected_indices = indices_for_label[:5]
#
#     # 添加到选定数据中
#     selected_data.append(data[selected_indices])
#
# # 将结果转换为 NumPy 数组
# selected_data = np.concatenate(selected_data, axis=0)
#
# # 打印结果
# print(selected_data)


# def replace_labels(samples_list, agg_label_xlsx):
#     """
#     替换标签
#     :param samples_list:
#     :param agg_label_xlsx:
#     :return:
#     """
#     #     print(samples_list)
#     df_train_20 = pd.DataFrame(samples_list, columns=["image", "label"])
#     df_true = pd.read_excel(agg_label_xlsx)
#     # """
#     images_filenames = []
#     for i in df_train_20["image"]:
#         i = i.split("\\")[-2:]
#         a = "\\".join(i)
#         images_filenames.append(a)
#     df_train_20["filenames"] = images_filenames
#     df = pd.merge(df_train_20, df_true, on="filenames", how='left')
#     # 后面真实操作的时候将true_label改成教师聚合label
#     df = get_noise_labels(epsilon, df)
#     # 使用教师聚合label和加噪后的label
#     df = df[["image_x", "teacher_labels"]]
#     samples = [list(x) for x in df.values]
#     targets = df["teacher_labels"].tolist()
#
#     return samples, targets
