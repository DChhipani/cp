%======================== Pre-Defined Variables ========================%
include "count.mzn";                                                    %

int:                       n;      % Number of nodes in G                %
int:                       T;      % Max time before game ends           %
array[int] of int:         f;      % Initial fire locations              %
array[1..n, 1..n] of 0..1: G;      % Adjacency matrix Graph              %
int:                       budget; % Number of firefighters placed per turn%

%=======================================================================%

% Time each node is defended
array[1..n] of var 0..T: d;   
% Time each node is burned
array[1..n] of var 0..T: b;   

% Constraint: Initial fire locations
constraint forall(i in f)(b[i] == 0);   % Initial fire locations at time 0
constraint count(b, 1, length(f));      % Ensure max fires at t=1
constraint count(d, 1, 0);              % No nodes defended at t=1

% Constraint: Cannot defend burning nodes, and cannot burn defended nodes
constraint forall(i in 1..n)(
    if b[i] != 0 then d[i] == 0     % Cannot defend burning nodes
    elseif d[i] != 0 then b[i] == 0 % Cannot burn defended nodes
    else true endif
);

% Constraint: Budget constraint (limit number of firefighters per time t)
constraint forall(t in 1..T)(
    count(j in d)(j == t) <= budget % Limit firefighters at each time t
);

% Fire Spread: If a node is burning and has neighbors that are burning, fire spreads
constraint forall(t in 2..T)(
    forall(x in 1..n)(
        if b[x] == t-1 then
            sum(y in 1..n where G[x, y] == 1 /\ b[y] == t-1)(1) > 0 % Fire spreads from burning neighbors
        else
            true
        endif
    )
);

% Defense Spread: If a node is defended at time t, it will not burn
constraint forall(t in 1..T)(
    forall(x in 1..n)(
        if d[x] == t then
            b[x] == 0 % Defended node cannot burn at time t
        endif
    )
);

% Fire spread based on adjacency matrix (neighbor interactions)
constraint forall(t in 2..T)(
    forall(x in 1..n)(
        forall(y in 1..n)(
            if G[x, y] == 1 /\ b[y] != 0 then
                if b[x] == 0 \/ b[x] > b[y] then
                    b[x] + d[x] == b[y] + 1 % Fire can spread from y to x
                endif
            endif
        )
    )
);

% Maximize the number of saved nodes (nodes not burned by the end)
solve maximize sum(i in 1..n)(if b[i] == 0 then 1 else 0 endif);
