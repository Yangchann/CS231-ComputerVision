# Thay thế dữ liệu trong biến data để tính giá trị trung bình của accuracy val
# Dưới đây là ví dụ dữ liệu của tổ hợp orientations: 9 và n_components: 50
data = """
kernel: linear, C: 0.01, accuracy train: 0.7187708750835003, accuracy val: 0.7003561887800535
kernel: linear, C: 0.1, accuracy train: 0.718198301364634, accuracy val: 0.6985752448797863
kernel: linear, C: 1, accuracy train: 0.7184845882240671, accuracy val: 0.699020480854853
kernel: linear, C: 10, accuracy train: 0.7182937303177784, accuracy val: 0.6985752448797863
kernel: poly, C: 0.01, accuracy train: 0.4503292298883481, accuracy val: 0.4385574354407836
kernel: poly, C: 0.1, accuracy train: 0.7127588510354042, accuracy val: 0.6540516473731077
kernel: poly, C: 1, accuracy train: 0.9303368642045997, accuracy val: 0.7475512021371327
kernel: poly, C: 10, accuracy train: 0.9991411394217006, accuracy val: 0.7373107747105966
kernel: rbf, C: 0.01, accuracy train: 0.584502338009352, accuracy val: 0.5801424755120214
kernel: rbf, C: 0.1, accuracy train: 0.7620956198110507, accuracy val: 0.7275155832591274
kernel: rbf, C: 1, accuracy train: 0.8935967172440118, accuracy val: 0.7666963490650045
kernel: rbf, C: 10, accuracy train: 0.9977097051245348, accuracy val: 0.7631344612644702
kernel: sigmoid, C: 0.01, accuracy train: 0.6797404332474473, accuracy val: 0.6749777382012466
kernel: sigmoid, C: 0.1, accuracy train: 0.6913827655310621, accuracy val: 0.6914514692787177
kernel: sigmoid, C: 1, accuracy train: 0.6088367210611699, accuracy val: 0.6322350845948352
kernel: sigmoid, C: 10, accuracy train: 0.5675159843496517, accuracy val: 0.5908281389136242
n_neighbors: 1, accuracy train: 0.9999045710468556, accuracy val: 0.691006233303651
n_neighbors: 2, accuracy train: 0.8299456054967077, accuracy val: 0.667853962600178
n_neighbors: 3, accuracy train: 0.8277507395743868, accuracy val: 0.691006233303651
n_neighbors: 4, accuracy train: 0.806660940929478, accuracy val: 0.7003561887800535
n_neighbors: 5, accuracy train: 0.7970226166618952, accuracy val: 0.715939447907391
n_neighbors: 6, accuracy train: 0.785762000190858, accuracy val: 0.7195013357079252
n_neighbors: 7, accuracy train: 0.7787956866113179, accuracy val: 0.7168299198575245
n_neighbors: 8, accuracy train: 0.7729745204695104, accuracy val: 0.7123775601068566
n_neighbors: 9, accuracy train: 0.7662944937494036, accuracy val: 0.7146037399821905
n_neighbors: 10, accuracy train: 0.767153354327703, accuracy val: 0.7128227960819234
n_neighbors: 11, accuracy train: 0.7654356331711041, accuracy val: 0.7088156723063224
n_neighbors: 12, accuracy train: 0.7612367592327512, accuracy val: 0.705253784505788
n_neighbors: 13, accuracy train: 0.7594236091230079, accuracy val: 0.7030276046304541
n_neighbors: 14, accuracy train: 0.7592327512167192, accuracy val: 0.7056990204808549
n_neighbors: 15, accuracy train: 0.756465311575532, accuracy val: 0.7092609082813891
n_neighbors: 16, accuracy train: 0.7544613035594999, accuracy val: 0.7101513802315227
n_neighbors: 17, accuracy train: 0.7538887298406336, accuracy val: 0.7083704363312555
n_neighbors: 18, accuracy train: 0.7504532875274358, accuracy val: 0.707479964381122
n_neighbors: 19, accuracy train: 0.7489264242771256, accuracy val: 0.7092609082813891
n_neighbors: 20, accuracy train: 0.7502624296211471, accuracy val: 0.7105966162065895
n_neighbors: 21, accuracy train: 0.749498997995992, accuracy val: 0.703472840605521
n_neighbors: 22, accuracy train: 0.7466361294016605, accuracy val: 0.697239536954586
n_neighbors: 23, accuracy train: 0.7478767057925375, accuracy val: 0.707479964381122
n_neighbors: 24, accuracy train: 0.7438686897604734, accuracy val: 0.7065894924309885
n_neighbors: 25, accuracy train: 0.7421509686038744, accuracy val: 0.7083704363312555
n_neighbors: 26, accuracy train: 0.7418646817444413, accuracy val: 0.7088156723063224
n_neighbors: 27, accuracy train: 0.7433915449947514, accuracy val: 0.7146037399821905
n_neighbors: 28, accuracy train: 0.740051531634698, accuracy val: 0.7083704363312555
n_neighbors: 29, accuracy train: 0.7397652447752648, accuracy val: 0.711487088156723
n_estimators: 100, accuracy train: 0.9999045710468556, accuracy val: 0.7235084594835263
"""

accuracy_vals = []
for line in data.split('\n'):
    if line.strip() != '':
        # kernel = line.split('kernel: ')[1].split(',')[0]
        # c = line.split('C: ')[1].split(',')[0]
        # if kernel == 'sigmoid' and float(c) == 0.01:
            accuracy_val = line.split('accuracy val: ')[1].split(',')[0]
            accuracy_vals.append(float(accuracy_val))
            print("accuracy val:", accuracy_val)


# Tính giá trị trung bình của accuracy val
average_accuracy_val = sum(accuracy_vals) / len(accuracy_vals)
print("Giá trị trung bình của accuracy val:", round(average_accuracy_val, 4))
