using RigidBodyDynamics
using MeshCat
using MeshCatMechanisms
using Random
using StaticArrays
using Rotations
using LinearAlgebra
using ForwardDiff
using GeometryBasics: HyperRectangle, Vec
using MeshCat: setobject!, MeshPhongMaterial, Translation, LinearMap
using Rotations
using Colors

const URDFPATH = joinpath(@__DIR__,"a1","urdf","a1.urdf")

function init_quadruped()
    a1 = parse_urdf(URDFPATH, floating=true, remove_fixed_tree_joints=false) 
    return a1
end

struct UnitreeA1{C}
    mech::Mechanism{Float64}
    statecache::C
    dyncache::DynamicsResultCache{Float64}
    xdot::Vector{Float64}
    function UnitreeA1(mech::Mechanism)
        nx = num_positions(mech) + num_velocities(mech)
        statecache = StateCache(mech)
        rescache = DynamicsResultCache(mech)
        xdot = zeros(nx)
        new{typeof(statecache)}(mech, statecache, rescache, xdot)
    end
end
function UnitreeA1()
    UnitreeA1(init_quadruped())
end

state_dim(model::UnitreeA1) = num_positions(model.mech) + num_velocities(model.mech)
control_dim(model::UnitreeA1) = 12 

function dynamics(model::UnitreeA1, x::AbstractVector{T1}, u::AbstractVector{T2}, λ::AbstractVector{T3}) where {T1,T2,T3} 
    T = promote_type(T1,T2,T3)
    state = model.statecache[T]
    res = model.dyncache[T]

    # Convert from state ordering to the ordering of the mechanism
    copyto!(state, x)
    q = configuration(state)
    jac1 = jac_foot(model, q, "RR")
    jac2 = jac_foot(model, q, "FR")
    jac3 = jac_foot(model, q, "FL")
    jac4 = jac_foot(model, q, "RL")
    τ = [zeros(6); u]
    dynamics!(res, state, τ + jac1'*λ[1:3] + jac2'*λ[4:6] + jac3'*λ[7:9] + jac4'*λ[10:12])
    q̇ = res.q̇
    v̇ = res.v̇
    return [q̇; v̇]
end

function initial_state(model::UnitreeA1)
    state = MechanismState(model.mech)
    a1 = model.mech
    zero!(state)
    leg = ("FR","FL","RR","RL")
    for i = 1:4
        # s = isodd(i) ? 1 : -1
        # f = i < 3 ? 1 : -1
        set_configuration!(state, findjoint(a1, leg[i] * "_hip_joint"), deg2rad(00))
        set_configuration!(state, findjoint(a1, leg[i] * "_thigh_joint"), deg2rad(50))
        set_configuration!(state, findjoint(a1, leg[i] * "_calf_joint"), deg2rad(-100))
    end
    floating_joint = findjoint(a1, "base_to_world")
    trunk_height = get_foot_position(model, configuration(state))[3]
    set_configuration!(state, floating_joint, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -trunk_height])

    return [configuration(state); velocity(state)]
end

function reference_trajectory(model::UnitreeA1, box_height=0.3, box_distance=0.7)

    tf = 0.5
    Δt = 1e-2
    t_vec = collect(0:Δt:tf)
    g = -9.81

    nu = control_dim(model)

    vx = box_distance/tf
    vz = (box_height - 0.5*g*tf^2)/tf #sqrt(box_height*2*g)
    @show vx
    @show vz
    v_des_vec = zeros(num_velocities(model.mech))
    v_des_vec[6] = vz
    v_des_vec[4] = vx

    xic = initial_state(model)
    state = MechanismState(model.mech)
    set_configuration!(state, xic[1:num_positions(model.mech)])
    set_velocity!(state, v_des_vec)
    ts, qs, vs = simulate(state, tf; Δt = Δt)

    N = length(t_vec) + 6
    t_extend = collect(Δt*(1:6)) .+ tf

    Xref = [[qs[i]; vs[i]] for i in 1:length(qs)]
    Xref = vcat(fill(first(Xref), 3), Xref, fill(last(Xref), 3))
    Uref = [[0.001*randn(nu); Δt] for i in 1:(N-1)]

    return Xref, Uref, [t_vec;t_extend]
end

# function goal_state(model::UnitreeA1)
#     state = model.statecache[Float64]
#     a1 = model.mech
#     zero!(state)
#     leg = ("FR","FL","RR","RL")
#     for i = 1:2
#         s = isodd(i) ? 1 : -1
#         f = i < 3 ? 1 : -1
#         # set FR, FL
#         set_configuration!(state, findjoint(a1, leg[i] * "_hip_joint"), -0.009836778663223645s)
#         set_configuration!(state, findjoint(a1, leg[i] * "_thigh_joint"), 0.016964453132483102)
#         set_configuration!(state, findjoint(a1, leg[i] * "_calf_joint"), 0.004758019542372259)

#         # set RR, RL 
#         set_configuration!(state, findjoint(a1, leg[i + 2] * "_hip_joint"), -0.0007997782745616991s)
#         set_configuration!(state, findjoint(a1, leg[i + 2] * "_thigh_joint"), 0.1272482316068922)
#         set_configuration!(state, findjoint(a1, leg[i + 2] * "_calf_joint"), 0.23113053507838607)
#     end
#     floating_joint = findjoint(a1, "base_to_world")
#     trunk_height = get_foot_position(model, configuration(state))[3]
#     # TODO: rotate trunk to appropriate pos
#     set_configuration!(state, floating_joint, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -trunk_height])

#     return [configuration(state); velocity(state)]
# end

##########################
##     FK FUNCTIONS     ## 
##########################

function get_trunk_position(model::UnitreeA1, q)
    mech = model.mech 
    T = eltype(q)
    state = MechanismState{T}(mech)
    set_configuration!(state, q)

    trunk = findbody(mech, "trunk")
    tf_world = transform_to_root(state, default_frame(trunk))
    # world = findbody(mech, "world") 
    # tf_world = relative_transform(state, default_frame(world), default_frame(trunk))
    return translation(tf_world)
end

function get_foot_position(model::UnitreeA1, q, foot="RR")
    mech = model.mech 
    T = eltype(q)
    state = MechanismState{T}(mech)
    set_configuration!(state, q)

    foot_body = findbody(mech, foot * "_foot")
    tf_world = transform_to_root(state, default_frame(foot_body))
    # world = findbody(mech, "world") 
    # tf_world = relative_transform(state, default_frame(world), default_frame(foot_body))
    return translation(tf_world)
end

function get_trunk_velocity(model::UnitreeA1, x)
    mech = model.mech 
    T = eltype(x)
    state = MechanismState{T}(mech)
    copyto!(state, x)
    trunk = findbody(mech, "trunk")
    twist = twist_wrt_world(state, trunk)
    v = linear(twist)
    return v
end

function jac_foot(model::UnitreeA1, q, foot="RR")
    mech = model.mech 
    T = eltype(q)
    state = MechanismState{T}(mech)
    set_configuration!(state, q)

    world_body = root_body(mech) #findbody(mech, "world")
    foot_body = findbody(mech, foot * "_foot")
    p = path(mech, world_body, foot_body)
    jac = geometric_jacobian(state, p)
    return jac.linear
end

##########################
## VISUALIZER FUNCTIONS ## 
##########################

function initialize_visualizer(a1::UnitreeA1)
    vis = Visualizer()
    delete!(vis)
    cd(joinpath(@__DIR__,"a1","urdf"))
    
    # Create robot visualization
    mvis = MechanismVisualizer(a1.mech, URDFVisuals(URDFPATH), vis)
    
    # Add block to the same visualizer
    add_block(vis)  # Using default parameters
    
    cd(@__DIR__)
    return mvis
end

function add_block(vis::Visualizer; 
    name::String="block",
    position=Translation(0.4, -0.3, 0.),  # Default position: 0.5m in front, centered
    size=Vec(1.0, 0.6, 0.3),              # 10cm cube
    color=RGB(0.8, 0.2, 0.2)              # Red color
)
    # Create and configure block geometry
    block = HyperRectangle(Vec(0.0, 0.0, 0.0), size)
    setobject!(vis[name], block, MeshPhongMaterial(color=color))
    
    # Apply position and orientation transformations
    settransform!(vis[name], position ∘ LinearMap(RotZ(0.0)))
end

##########################
##        UTILS         ## 
##########################

function create_idx(nx,nu,nλ,N)
    # This function creates some useful indexing tools for Z 
    # x_i = Z[idx.x[i]]
    # u_i = Z[idx.u[i]]    
    
    # our Z vector is [x0, u0, x1, u1, …, xN]
    nz = (N-1) * nu + (N-1) * nλ + N * nx # length of Z 
    x = [(i - 1) * (nx + nu + nλ) .+ (1 : nx) for i = 1:N]
    u = [(i - 1) * (nx + nu + nλ) .+ ((nx + 1):(nx + nu)) for i = 1:(N - 1)]
    λ = [(i - 1) * (nx + nu + nλ) .+ ((nx + nu + 1):(nx + nu + nλ)) for i = 1:(N - 1)]
    
    # constraint indexing for the (N-1) dynamics constraints when stacked up
    c = [(i - 1) * (nx) .+ (1 : nx) for i = 1:(N - 1)]
    nc = (N - 1) * nx # (N-1)*nx 
    
    return (nx=nx,nu=nu,nλ=nλ, N=N,nz=nz,nc=nc,x= x,u = u,λ=λ, c = c)
end

function quat2mrp(q)
    # Convert quaternion to modified Rodrigues parameters (MRP)
    q = QuatRotation(q)
    axis_angle = AngleAxis(q)

    theta = axis_angle.theta
    axis = [axis_angle.axis_x; axis_angle.axis_y; axis_angle.axis_z]

    return theta, axis
end

function mrp2quat(theta, axis)
    # Convert modified Rodrigues parameters (MRP) to quaternion
    aa = AngleAxis(theta, axis...)
    q = QuatRotation(aa)

    return [q.q.s ; q.q.v1; q.q.v2; q.q.v3]
end