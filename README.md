Example program showing how to render frames with Vulkan on Nvidia GPU and
present them to wayland surface through Intel GPU even when compositor doesn't
support Nvidia (e.g. Sway).

Made as a way to grasp some intuitive understanding of Vulkan api. Theoretically
it should be possible to implement something like this as Vulkan layer and allow
all Vulkan apps run seamlessly (with a price of additional buffers), but the
amount of necessary bookkeeping is enormous and whatever. It's just easier to
buy Radeon.
