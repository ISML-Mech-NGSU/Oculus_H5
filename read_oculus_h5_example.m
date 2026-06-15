%% read_oculus_h5_example.m
clear; clc; close all;

filename = 'output.h5';

%% HDF5の中身を確認
h5disp(filename)

%% データセット一覧
info = h5info(filename);
disp({info.Datasets.Name}')

%% 表示するフレーム番号
k = 2;

%% メタデータ読み込み
bearingCount    = h5read(filename, '/bearingCount');
rangeCount      = h5read(filename, '/rangeCount');
rangeResolution = h5read(filename, '/rangeResolution');
pingId          = h5read(filename, '/pingId');
frequency       = h5read(filename, '/frequency');

%% k番目フレームの画像を読む
% /images のサイズは frame x bearing x range
I = h5read(filename, '/images', ...
           [k 1 1], ...
           [1 Inf Inf]);

I = squeeze(I);   % bearing x range

%% k番目フレームの有効サイズ
nb = double(bearingCount(k));
nr = double(rangeCount(k));
dr = rangeResolution(k);

I = I(1:nb, 1:nr);

%% 方位角 [deg] を読む
bearing = h5read(filename, '/bearingsDeg', ...
                 [k 1], ...
                 [1 nb]);

bearing = squeeze(bearing);

%% 距離軸 [m]
range = (0:nr-1) * dr;

%% 表示：極座標画像のまま
figure;
imagesc(bearing, range, I.');
axis xy;
axis image;
colormap gray;
xlabel('Bearing [deg]');
ylabel('Range [m]');
title(sprintf('Oculus frame %d, pingId %d, %.1f MHz', ...
    k, pingId(k), frequency(k)/1e6));

%% 簡単なフィルタ例：メディアンフィルタ
If = medfilt2(I, [3 3]);

figure;
imagesc(bearing, range, If.');
axis xy;
axis image;
colormap gray;
xlabel('Bearing [deg]');
ylabel('Range [m]');
title('Median filtered image');

%% MATファイルとして保存したい場合
oculus.filename = filename;
oculus.image = I;
oculus.imageFiltered = If;
oculus.bearingDeg = bearing;
oculus.rangeM = range;
oculus.pingId = pingId(k);
oculus.frequency = frequency(k);
oculus.rangeResolution = dr;

save('oculus_one_frame.mat', 'oculus');