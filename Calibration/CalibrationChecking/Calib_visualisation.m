function Calib_visualisation(dirIn,Calib_path,CamID,Nplanes)

%% Plot the calibration points on the raw image in order to check if the calibration is good 
%% Input 
% dirIn     : the directory where the raw image and the calib file are
% Nplane    : the number of the plane you want to check
%% Output 
% N figure (where N is the number of cameras) where the cross are the point
% you place during the claibration
%The idea is to check that the point are centered on the calibration plate
% /!\ It's really usefull to check because bad calibration=bad result
%% 


%%Load the calibration and definition of pimg
% calib_path=sprintf('%s/calib2D_%d_cam%d.mat',dirIn, Nplane, CamID);
% calib = load(calib_path);
A = open(Calib_path);
calib = A.calib;

%%Load the image file
for z=Nplanes
    for kcam=CamID
        kcam
        PimgX=calib(z,kcam).pimg(:,1);  % x positions
        PimgY=calib(z,kcam).pimg(:,2);  % y positions
        figure('numberTitle','off','Name',sprintf('Cam %d',kcam))
        filename = sprintf('%s/MyCalibration_cam%d_%03d.%s',dirIn, kcam, z, 'tif');
        Img=imread(filename);
        imagesc(Img); colormap(gray);
        hold on
        plot(PimgX,PimgY,'o');
        fname = ['Plane ' num2str(z) ', camera ' num2str(kcam)];
        title(fname);
        PimgX=[];
        PimgY=[];
        filename=[];
    end
end 
