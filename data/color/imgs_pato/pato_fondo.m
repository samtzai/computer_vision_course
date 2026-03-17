clc;
I1 =  im2double(imread('Pato_01.jpg'));
I2 =  im2double(imread('Pato_02.jpg'));
I3 =  im2double(imread('Pato_03.jpg'));
I4 =  im2double(imread('Pato_04.jpg'));
I5 =  im2double(imread('Pato_05.jpg'));

img_fondo = I1 + I2 + I3 + I4 + I5;
img_fondo = img_fondo ./ 5;
imshow(img_fondo);

iframe =im2double(imread('Pato_10.jpg'));

imvto = imabsdiff(I1, iframe);
imshow(imvto);
imwrite(imvto, 'patomovto.jpg');