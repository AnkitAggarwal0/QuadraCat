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
using Rotations: RotZ
using Colors


const URDFPATH = joinpath(@__DIR__,"a1","urdf","a1.urdf")
# const URDFPATH = "/Users/kevintracy/devel/hw_ideas/hw1/Q2/a1/urdf/a1.urdf"

function has_joint(mech::Mechanism, joint_name::String)
    try
        joint = findjoint(mech, joint_name)
        return true
    catch e
        return false
    end
end


function attach_foot!(mech::Mechanism{T}, foot_name::String="RR"; revolute::Bool=true) where T
    # Get the relevant bodies from the model
    foot = findbody(mech, foot_name * "_foot")
    trunk = findbody(mech, "trunk")
    world = findbody(mech, "world")

    # Get the location of the foot
    state = MechanismState(mech)
    trunk_to_foot = translation(relative_transform(state, 
        default_frame(trunk), default_frame(foot)))
    foot_location = SA[trunk_to_foot[1], trunk_to_foot[2], 0]  # set the z height to 0

    # Build the new foot joint
    if !revolute 
        foot_joint = Joint("foot_joint", QuaternionSpherical{T}())
        world_to_joint = Transform3D(
            frame_before(foot_joint),
            default_frame(world),
            -foot_location        
        )

        # Attach to model
        attach!(mech,
            world,
            foot,
            foot_joint,
            joint_pose = world_to_joint,
        )
        if has_joint(mech, "base_to_world")
            remove_joint!(mech, findjoint(mech, "base_to_world"))
        end
    else
        # Create dummy bodies 
        dummy1 = RigidBody{T}("dummy1_" * foot_name)
        dummy2 = RigidBody{T}("dummy2_" * foot_name)
        for body ∈ (dummy1, dummy2)
            inertia = SpatialInertia(default_frame(body),
                moment = I(3)*1e-3,
                com    = SA[0,0,0],
                mass   = 1e-3
            )
            spatial_inertia!(body, inertia)
        end

        # X-Joint
        foot_joint_x = Joint("foot_joint_x_" * foot_name, Revolute{T}(SA[1,0,0]))
        world_to_joint = Transform3D(
            frame_before(foot_joint_x),
            default_frame(world),
            -foot_location        
        )
        attach!(mech,
            world,
            dummy1,
            foot_joint_x,
            joint_pose = world_to_joint
        )

        # Y-Joint
        foot_joint_y = Joint("foot_joint_y_" * foot_name, Revolute{T}(SA[0,1,0]))
        dummy_to_dummy = Transform3D(
            frame_before(foot_joint_y),
            default_frame(dummy1),
            SA[0,0,0]
        )
        attach!(mech,
            dummy1,
            dummy2,
            foot_joint_y,
            joint_pose = dummy_to_dummy 
        )

        # Z-Joint
        foot_joint_z = Joint("foot_joint_z_" * foot_name, Revolute{T}(SA[0,0,1]))
        joint_to_foot = Transform3D(
            frame_before(foot_joint_z),
            default_frame(dummy2),
            SA[0,0,0]
        )
        attach!(mech,
            dummy2,
            foot,
            foot_joint_z,
            joint_pose = joint_to_foot
        )
        if has_joint(mech, "base_to_world")
            remove_joint!(mech, findjoint(mech, "base_to_world"))
        end
        
    end
end

function build_quadruped()
    a1 = parse_urdf(URDFPATH, floating=true, remove_fixed_tree_joints=false) 
    attach_foot!(a1)
    # for leg in ("FR", "RL", "RR", "FL")
    #     attach_foot!(a1, leg; revolute=true)
    # end
    
    return a1
end

struct UnitreeA1{C}
    mech::Mechanism{Float64}
    statecache::C
    dyncache::DynamicsResultCache{Float64}
    xdot::Vector{Float64}
    function UnitreeA1(mech::Mechanism)
        N = num_positions(mech) + num_velocities(mech)
        statecache = StateCache(mech)
        rescache = DynamicsResultCache(mech)
        xdot = zeros(N)
        new{typeof(statecache)}(mech, statecache, rescache, xdot)
    end
end
function UnitreeA1()
    UnitreeA1(build_quadruped())
end

state_dim(model::UnitreeA1) = 30
control_dim(model::UnitreeA1) = 12 
function get_partition(model::UnitreeA1)
    n,m = state_dim(model), control_dim(model)
    return 1:n, n .+ (1:m), n+m .+ (1:n)
end

function dynamics(model::UnitreeA1, x::AbstractVector{T1}, u::AbstractVector{T2}) where {T1,T2} 
    T = promote_type(T1,T2)
    state = model.statecache[T]
    res = model.dyncache[T]

    # Convert from state ordering to the ordering of the mechanism
    copyto!(state, x)
    τ = [zeros(3); u]
    dynamics!(res, state, τ)
    q̇ = res.q̇
    v̇ = res.v̇
    return [q̇; v̇]
end

function jacobian(model::UnitreeA1, x, u)
    ix = SVector{30}(1:30)
    iu = SVector{12}(31:42)
    faug(z) = dynamics(model, z[ix], z[iu])
    z = [x; u]
    ForwardDiff.jacobian(faug, z)
end

# Set initial guess
function initial_state(model::UnitreeA1)
    state = model.statecache[Float64]
    a1 = model.mech
    zero!(state)
    # leg = ("FR","FL","RR","RL")
    # for i = 1:4
    #     s = isodd(i) ? 1 : -1
    #     f = i < 3 ? 1 : -1
    #     set_configuration!(state, findjoint(a1, leg[i] * "_hip_joint"), deg2rad(-20s))
    #     set_configuration!(state, findjoint(a1, leg[i] * "_thigh_joint"), deg2rad(-30f))
    #     set_configuration!(state, findjoint(a1, leg[i] * "_calf_joint"), deg2rad(10f))
    # end
    # set_configuration!(state, findjoint(a1, "foot_joint_x"), deg2rad(00))
    # set_configuration!(state, findjoint(a1, "foot_joint_y"), deg2rad(-00))

    return [configuration(state); velocity(state)]
end

function goal_state(model::UnitreeA1, box_height::Float64, box_distance::Float64)
    state = model.statecache[Float64]
    a1 = model.mech
    zero!(state)
    leg = ("FR", "FL", "RR", "RL")
    for i = 1:4
        s = isodd(i) ? 1 : -1  
        f = i < 3 ? 1 : -1     

        # Adjust joint angles for landing posture
        set_configuration!(state, findjoint(a1, leg[i] * "_hip_joint"), deg2rad(0))
        set_configuration!(state, findjoint(a1, leg[i] * "_thigh_joint"), deg2rad(-45f))
        set_configuration!(state, findjoint(a1, leg[i] * "_calf_joint"), deg2rad(45f))
    end

    # Set base position to be on top of the box
    set_configuration!(state, findjoint(a1, "floating_base"), [box_distance, 0.0, box_height])

    # Set velocities to zero (robot is stationary after landing)
    zero_velocity!(state)

    return [configuration(state); velocity(state)]
end

function switch_to_aerial!(mech::Mechanism, foot="RR")
    # remove foot constraint
    detach_foot!(mech, foot; revolute=true)
    # re-add floating base
    add_floating_base!(mech)
end

function detach_foot!(mech::Mechanism, foot="RR"; revolute=true)
    if revolute
        for axis in ["x", "y", "z"]
            name = "foot_joint_$axis"
            joint = findjoint(mech, name)
            joint !== nothing && remove_joint!(mech, joint)
        end
    else
        joint = findjoint(mech, "foot_joint")
        joint !== nothing && remove_joint!(mech, joint)
    end
end

function add_floating_base!(mech::Mechanism)
    base = findbody(mech, "trunk")
    world = findbody(mech, "world")

    floating_joint = Joint("base_to_world", Floating{Float64}())
    attach!(mech, world, base, floating_joint)
end


# function initialize_visualizer(a1::UnitreeA1)
#     vis = Visualizer()
#     delete!(vis)
#     cd(joinpath(@__DIR__,"a1","urdf"))
#     mvis = MechanismVisuusing GeometryBasics: HyperRectanglealizer(a1.mech, URDFVisuals(URDFPATH), vis)
#     cd(@__DIR__)
#     return mvis
# end



function add_block(vis::Visualizer; 
    name::String="block",
    position=Translation(0.5, -0.3, 0.),  # Default position: 0.5m in front, centered
    size=Vec(1.0, 0.6, 0.3),              # 10cm cube
    color=RGB(0.8, 0.2, 0.2)              # Red color
)
    # Create and configure block geometry
    block = HyperRectangle(Vec(0.0, 0.0, 0.0), size)
    setobject!(vis[name], block, MeshPhongMaterial(color=color))
    
    # Apply position and orientation transformations
    settransform!(vis[name], position ∘ LinearMap(RotZ(0.0)))
end

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

# function visualize_trajectory(model::UnitreeA1, mvis::Visualizer, trajectory::Matrix)
#     for t in eachcol(trajectory)
#         set_configuration!(mvis, t[1:state_dim(model)÷2])
#         render(mvis)
#         sleep(0.05)
#     end
# end
#visualize_trajectory(model, mvis, trajectory_solution)


# a1 = UnitreeA1()
# mvis = initialize_visualizer(a1)
# open(mvis)
