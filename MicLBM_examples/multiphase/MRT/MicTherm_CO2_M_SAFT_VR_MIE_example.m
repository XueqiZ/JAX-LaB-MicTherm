%% CO2 M-SAFT-VR-Mie UserProperties API example
% Calculates pressure, viscosity, and surface tension on a user-defined
% temperature-density grid.

clc;
clear;
close all;

exampleFolder = fileparts(mfilename('fullpath'));
trunkFolder = fileparts(fileparts(exampleFolder));
addpath(trunkFolder);
addpath(genpath(fullfile(trunkFolder,'src')));

%% CO2 MicTherm settings
Units           = 'Units=SI_reduced';
Components      = 'N_components=1';
EOSmodel        = 'EOS=M_SAFT_VR_MIE';
Output          = 'Output=no';
Debug           = 'Debug=no';
Stability       = 'stability=no';
SubstanceID1    = 'Substance_ID1=0';
PotModel1       = 'PotModel_1=LJ.pm';
IdealModel      = 'IDEAL=IdealQM';
Epsilon1        = 'epsilon_1=182.32';
Sigma1          = 'sigma_1=3.7131';
Chainlength1    = 'chainlength=1.0677';
MolarMass1      = 'molar_mass_1=44.009';
CASNumber1      = 'CAS_number_1=124-38-9';
Quadrupole1     = 'Quadrupolemoment_1=4.4';
NQuadrupole1    = 'N_Quadrupolemoment_1=1';
EtaParameter    = 'eta_param=0,-2.5399,2.9133,-0.2182,0.1340';
DGTKappa1      = 'DGT_kappa_1=2.5726';
TransportMode   = 'transPropMode=entropyScaling';
Properties      = 'properties=p, eta, gamma_surface';

BaseSettings = {Units, Components, EOSmodel, Output, ...
    Debug, Stability, SubstanceID1, PotModel1, IdealModel, Epsilon1, ...
    Sigma1, Chainlength1, MolarMass1, CASNumber1, Quadrupole1, ...
    NQuadrupole1, EtaParameter, DGTKappa1, TransportMode};

%% Critical point
[CriticalName,CriticalUnit,CriticalValue] = API_example( ...
    'uninitialized',[],[],[],[], ...
    [BaseSettings, {'calculationmode = critical_point'}]);

Tc = CriticalValue(1,1);
rhoc = CriticalValue(1,2);
pc = CriticalValue(1,3);

fprintf('CO2 critical point: Tc = %.8g, rhoc = %.8g, pc = %.8g\n', ...
    Tc,rhoc,pc);
disp('Critical-point output names:')
disp(CriticalName)
disp('Critical-point output units:')
disp(CriticalUnit)

%% Full vapor-liquid equilibrium curve
[VLEName,VLEUnit,VLEValue] = API_example( ...
    'uninitialized',[],[],[],1, ...
    [BaseSettings, {'calculationmode = phaseEquilib', 'dT = 0.1'}]);

VLE_T = VLEValue(:,1);
VLE_rho_l = VLEValue(:,2);
VLE_rho_g = VLEValue(:,3);
VLE_p = VLEValue(:,4);
spinrho_l = VLEValue(:,6);
spinrho_g = VLEValue(:,7);
spinp_l = VLEValue(:,8);
spinp_g = VLEValue(:,9);

validVLE = isfinite(VLE_T) & isfinite(VLE_rho_l) & isfinite(VLE_rho_g);
VLE_T = VLE_T(validVLE);
VLE_rho_l = VLE_rho_l(validVLE);
VLE_rho_g = VLE_rho_g(validVLE);
VLE_p = VLE_p(validVLE);
VLE_p_l = VLE_p;
VLE_p_g = VLE_p;
spinrho_l = spinrho_l(validVLE);
spinrho_g = spinrho_g(validVLE);
spinp_l = spinp_l(validVLE);
spinp_g = spinp_g(validVLE);

if numel(VLE_T) < 2
    error('MicTherm returned fewer than two valid VLE points.');
end

disp('VLE output names:')
disp(VLEName)
disp('VLE output units:')
disp(VLEUnit)

%% UserProperties calculation
temperatureValues = (250:10:320)';
densityValues = (0.1:0.5:20)';
[rhoGrid,TGrid] = meshgrid(densityValues,temperatureValues);

TInput = TGrid(:);
rhoInput = rhoGrid(:);
xInput = ones(size(TInput));

[PropertyName,PropertyUnit,PropertyValue] = API_example( ...
    'uninitialized',rhoInput,TInput,[],xInput, ...
    [BaseSettings, {'APIMode=UserProperties',Properties}]);

propertyColumn = 2;
pGrid = reshape(PropertyValue(:,propertyColumn+0),size(TGrid));
etaGrid = reshape(PropertyValue(:,propertyColumn+1),size(TGrid));
gammaGrid = reshape(PropertyValue(:,propertyColumn+2),size(TGrid));

fprintf('Calculated %d CO2 UserProperties states.\n',numel(TInput));
disp('User-property output names:')
disp(PropertyName)
disp('User-property output units:')
disp(PropertyUnit)

%% Plots
figure('Color','w','Name','MicTherm CO2 properties');

subplot(1,3,1)
surf(rhoGrid,TGrid,pGrid,'EdgeColor','none')
title('Pressure')
xlabel('\rho / mol L^{-1}')
ylabel('T / K')
zlabel('p')
colorbar
view(135,30)

subplot(1,3,2)
surf(rhoGrid,TGrid,etaGrid,'EdgeColor','none')
title('Viscosity')
xlabel('\rho / mol L^{-1}')
ylabel('T / K')
zlabel('\eta')
colorbar
view(135,30)

subplot(1,3,3)
surf(rhoGrid,TGrid,gammaGrid,'EdgeColor','none')
title('Surface tension')
xlabel('\rho / mol L^{-1}')
ylabel('T / K')
zlabel('\gamma_{surface}')
colorbar
view(135,30)

sgtitle('CO2 M-SAFT-VR-Mie UserProperties from MicTherm')

%% VLE coexistence plots
figure('Color','w','Name','MicTherm CO2 VLE');
plot3(VLE_T,VLE_rho_l,VLE_p_l,'o-','LineWidth',1.4, ...
    'DisplayName','Liquid')
hold on
plot3(VLE_T,VLE_rho_g,VLE_p_g,'s-','LineWidth',1.4, ...
    'DisplayName','Vapor')
grid on
box on
xlabel('T')
ylabel('\rho')
zlabel('p')
title('VLE: T-\rho-p')
legend('Location','best')
view(135,30)


resultsFolder = fullfile(exampleFolder,'results');
if ~isfolder(resultsFolder), mkdir(resultsFolder); end
save(fullfile(resultsFolder,'CO2_MicTherm_grid.mat'), ...
    'Tc','rhoc','pc','VLE_T','VLE_rho_l','VLE_rho_g', ...
    'VLE_p_l','VLE_p_g', ...
    'TGrid','rhoGrid','pGrid','etaGrid','gammaGrid', ...
    'spinrho_g','spinp_g','spinrho_l','spinp_l');
