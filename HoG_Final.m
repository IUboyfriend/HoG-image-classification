% nr_b = 2;
% nc_b = 2;
nr_b = 3;
nc_b = 3;
nbin=9;

Training_Hog = zeros(5, 5, nbin*nc_b*nr_b);%store all the traning images' Hog
Testing_Hog = zeros(5, 5, nbin*nc_b*nr_b);%store all the testing images' Hog

%calculate the training samples' Hog
for classNo = 1:5
    for imageNo = 1:5
         Training_Hog(classNo, imageNo, : ) = HoG_Result(classNo, imageNo,nr_b,nc_b,nbin, 0);
    end
end

for classNo = 1:5
    for imageNo = 1:5
         Testing_Hog(classNo, imageNo, : ) = HoG_Result(classNo, imageNo,nr_b,nc_b,nbin, 1);
    end
end

correct = 0;

for classNo_test = 1:5
    for imageNo_test = 1:5
        class = 0;
        difference = 1000000;
        for classNo_train = 1:5
            for imageNo_train = 1:5
                % % L1 metric
                current = abs(Testing_Hog(classNo_test,imageNo_test,:) - Training_Hog(classNo_train,imageNo_train,:));

                % L2 metric
                % current =(Testing_Hog(classNo_test,imageNo_test,:) - Training_Hog(classNo_train,imageNo_train,:)) .^2;

                % Chi-square
                % a = (Testing_Hog(classNo_test,imageNo_test,:) - Training_Hog(classNo_train,imageNo_train,:)) .^2;
                % b = Testing_Hog(classNo_test,imageNo_test,:) + Training_Hog(classNo_train,imageNo_train,:);
                % current = a ./ b;

                current_sum = sum (current);
                if current_sum < difference
                    class = classNo_train;
                    difference = current_sum;
                end
            end
        end

        if class == classNo_test
            correct = correct + 1;
        end

    end
end

accuracy = correct / 25;

function [Image_HoG] = HoG_Result (class_no, image_no, nr_b, nc_b, nbin, flag )

    if flag == 0
        suffix = '_Training.bmp';
    else
        suffix = '_Test.bmp';
    end

    filepath = strcat(num2str(class_no), '/', num2str(class_no), num2str(image_no),suffix);
    A = imread(filepath);
    A_gray = im2gray(A);
    [nr, nc] = size(A_gray);
    Sx=[-1 0 1; -2 0 2; -1 0 1]; % Sobel operator
    Sy=[-1 -2 -1; 0 0 0; 1 2 1]; % Sobel operator
    Ax=imfilter(double(A_gray), Sx);
    Ay=imfilter(double(A_gray), Sy);
              
    I_mag=sqrt(Ax.^2+ Ay.^2);% I_mag: gradient magnitude
    % I_angle: gradient orientation
      for j=1:nr
        for i=1:nc
          if abs(Ax(j, i))<=0.0001 & abs(Ay(j, i))<=0.0001 % Both Ix and Iy are close to zero
            I_angle(j, i) = 0.00;
          else
            Ipr(j, i) = atan(Ay(j,i)/Ax(j,i)); % Compute the angle in radians
            I_angle(j, i) = Ipr(j, i)*180/pi; % Compute the angle in degrees
            if I_angle(j, i) < 0 % If the angle is negative, 180 degrees added
               I_angle(j, i)=180+I_angle(j, i);
            end
          end
        end
      end
    
    nr_size = nr/nr_b;
    nc_size = nc/nc_b;
    Image_HoG = zeros(1, nbin*nr_b*nc_b);
    for i=1:nr_b
      for j=1:nc_b
    %extract the magnitude and gradient of a block out
        I_mag_block = I_mag((i-1)*nr_size+1:i*nr_size, (j-1)*nc_size+1:j*nc_size);
        I_angle_block = I_angle((i-1)*nr_size+1:i*nr_size, (j-1)*nc_size+1:j*nc_size);
        % HoG1 is a function which create the HoG histogram
        gh=HoG1(I_mag_block, I_angle_block, nbin);
        % Histogram_Normalization is a function to normalize the input histogram gh
        ngh=Histogram_Normalization(gh);
    
        pos = (j-1)*nbin+(i-1)*nc_b*nbin+1; %put the left column first and then the right column
        Image_HoG(pos:pos+nbin-1) = ngh;
      end
    end

end


function [ ghist ] = HoG1( Im, Ip, nbin )
  % Compute the HoG of an image block, with unsigned gradient (i.e. 0-180)
  % Im: magnitude of the image block
  % Ip: orientation of the image block
  % nbin: number of orientation bins
  ghist = zeros(1, nbin);
  [nr1 nc1] = size(Im);
  % Compute the HoG
  interval = 20; % the interval for a bin, and what is ????”?
  for i = 1:nr1
    for j = 1:nc1
      index = floor(Ip(i, j)/interval)+1;
      ghist(index) = ghist(index) + Im(i,j); % what is  ????g?
    end
  end
end

function [ nhist ] = Histogram_Normalization( ihist )
  % Normalize input histogram ihist to a unit histogram
  total_sum = sum(ihist);
  nhist = ihist / total_sum; % what is ????
end

