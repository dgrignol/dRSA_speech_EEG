function [SSIndices SSIndicesPlot] = dRSA_subsampling_v3(maskSubsampling, opt)
% get start and end points of subsamples)

% input args:
%
% maskSubsampling = 1xN logical vector with 1 = masked out time points (final subsamples should not overlap with TPs = 1)
% maskSubsamplingStarts = 1xN logical vector with 1 = masked out time points (final subsamples should not START WITH TPs = 1. Otherwise it can happen that the found subsample overlaps with masked TPs)
%
% opt = struct array with:
% opt.nSubSamples (how many subsamples should be selected from Event)
% opt.SubSampleDur (how many time points per subsample. nSubSamples*SubSampleDur must be smaller than number of time points
% opt.nIter (how many subsampling iterations. If opt.SubSampleDur ==  size(KxLxMxN,1): no subsampling, e.g. 14 5s-long trials, dRSA on the full 5s)
%
% opt.spacing (minimum distance (num of time points) between subsamples. Default = 1)
%
% output: 
% SSIndices = 3D array with dimension: 1: num of subsamples, 2: num of TPs per subsample, and 3: num of iterations
% SSIndicesPlot = 1xN logical vector with 1 = found subsamples

SSIndices = [];
SeqIndicesPlot=[];
fprintf('create subsamples:      ')

breakloop = 0;

for i = 1:opt.nIter
    fprintf('\b\b\b\b\b%04d ',i);
    % select random resampling trials, that should be spaced at least x secs
    
     if breakloop == 1
               disp('Totally no solution found');
               break;
     end
    
    % original maskSubsamplingStarts stays the same
    maskSubsamplingTotal = maskSubsampling;
    constraints=0;

    while constraints == 0
        sample = 1;
        SSstarts = [];
        newSampleRestart = 0;
        gatekeepeingvariable = 0;
        totalrestart = 0;
       fprintf('\n placing subsamples:      ')

       while sample <= opt.nSubSamples  
           if totalrestart > 500
               breakloop = 1;
               disp('No solution found');
               break;
               
           elseif newSampleRestart > 500
               sample = 1;
               SSstarts = []; %empty the cache
               maskSubsamplingTotal = maskSubsampling; %reset the mask
               newSampleRestart = 0;
               totalrestart = totalrestart +1;  %if we restart too often, then we break it to avoid an infinite loop.
               gatekeepeingvariable = 0;
               
           else
               try
                   % start positions of subsamples
                   SSstart = sort(randsample(find(maskSubsamplingTotal==0),1));
                   gatekeepeingvariable = 1;
                   
               catch   %if we find no solution, maybe the first Subsample was not placed well, so lets try again
                   sample = 1;
                   SSstarts = []; %empty the cache
                   maskSubsamplingTotal = maskSubsamplingStarts; %reset the mask
                   totalrestart = totalrestart +1;  %if we restart too often, then we break it to avoid an infinite loop.
                   gatekeepeingvariable = 0;
                   
               end
               
               if gatekeepeingvariable  %if we are in the try condition
                   % end point of samples
                   SSend = SSstart+opt.SubSampleDur + opt.spacing;  %end point is the start + Duration + Minumum Spacing between it
                   
                   if SSend <= size(maskSubsamplingTotal, 2) && sum( maskSubsamplingTotal(SSstart:SSend) ) == 0  %in case that the start point is not placed, but not everything.
                       maskSubsamplingTotal(SSstart:SSend) = 1;  % so now we can't draw a second time from here
                       SSstarts = [SSstarts, SSstart];
                       newSampleRestart = 0;
                      
                       fprintf('\b\b\b\b\b%04d ',sample);
                       sample = sample +1; % get to next sample

                   else
                       %sample = sample - 1;  %jump back one sample and try again
                       newSampleRestart = newSampleRestart+1;
                   end
               end
               
               
           end
       end
        
        if totalrestart > 500
             breakloop = 1;
            disp('No solution found');
            break;
        end
        
        SSstarts = sort(SSstarts);
        SSends = SSstarts + opt.SubSampleDur;
        
        
        
        %identify the gaps
        gaps = SSstarts(2:end) - SSends(1:end-1);
       
        % compute all indices for each subsample
        idx = (repmat(SSstarts,opt.SubSampleDur,1) + (0:opt.SubSampleDur-1)')'; % conditions x frames x iterations

        if sum(ismember(idx(:),find(maskSubsampling)))==0 && ... % subsamples are within mask
           min(gaps)>=opt.spacing && ... % there is sufficient gap between subsamples
           max(idx(:))<numel(maskSubsampling) % there are enough TPs for all subsamples
           % old: numel(unique(idx))==opt.nSubSamples*opt.SubSampleDur && ... % subsamples do not overlap with each other

            SSIndices(:,:,i) = idx;
            constraints=1;

            % for plotting:
            SSIndicesPlot(i,:) = zeros(size(maskSubsampling));
            SSIndicesPlot(i,maskSubsampling) = 1;
            SSIndicesPlot(i,idx(:)) = 2;

        end
    end
    %Find_SubsampleSpaces (maskSubsamplingTotal, opt);  %for graphics
end

end