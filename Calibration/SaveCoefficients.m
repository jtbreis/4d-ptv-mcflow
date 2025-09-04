function [coeffs_trans] = SaveCoefficients(dirIn,CalibPath,planes,Z,camID)
%% Charger la calib
fprintf("Loading calibration file... \n")
load(CalibPath,'calib');

% Initialisation struture
coeffs_trans(numel(planes), numel(camID)).posPlane = [];
coeffs_trans(numel(planes), numel(camID)).T3rw2px = [];
coeffs_trans(numel(planes), numel(camID)).T3px2rw = [];
coeffs_trans(numel(planes), numel(camID)).T1px2rw = [];

fprintf("Let's start loop over cameras...")
% Loop over cameras
for kcam=1:numel(camID)
    kcam
    for kz=1:numel(planes)
        kz
        T3rw2px = calib(kz,kcam).T3rw2px;
        T3px2rw = calib(kz,kcam).T3px2rw;
        T1px2rw = calib(kz,kcam).T1px2rw;
        
        coeffs_trans(kz,kcam).posPlane = Z(kz);
        coeffs_trans(kz,kcam).T3rw2px = [T3rw2px.A' T3rw2px.B'];
        coeffs_trans(kz,kcam).T3px2rw = [T3px2rw.A' T3px2rw.B'];
        coeffs_trans(kz,kcam).T1px2rw = [T1px2rw.T];
    end
end

save(sprintf('%s/coefficients_transformation.mat',dirIn),'coeffs_trans');
end