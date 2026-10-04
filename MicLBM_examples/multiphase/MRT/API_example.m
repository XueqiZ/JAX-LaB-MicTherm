function [ Name , Units , Value ] = API_example( initMode , rho , T , p , x , userParameters )

    % example skript for MicTherm API
    % initialize API 
    [ Name , Units , Value ]= com_API(initMode,rho,T,p,x,userParameters{:});
    if initMode == "uninitialized"
        % % calculate state points for given Tpx (or rhoTx) 
        [ Name , Units , Value ] = com_API('initialized',rho,T,p,x);
    end

end