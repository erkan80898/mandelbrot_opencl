use ggez::conf;
use ggez::event::{self, EventHandler};
use ggez::graphics;
use ggez::graphics::{Canvas, Color, DrawParam, Drawable, Image};
use ggez::input::mouse::MouseButton;
use ggez::{Context, ContextBuilder, GameResult};
use image::{Rgba};
use ocl::{
    Buffer, Context as ContextOCL, Device, Kernel, MemFlags, Platform, Program, Queue,
};
use palette::gradient::Gradient;
use palette::rgb::LinSrgb;
const MAX_ITER: u32 = 256;

static KERNEL_SRC: &'static str = r#"
    __kernel void mandelbrot(__global unsigned char colors[][4],
                                    int width, int height,int iter_limit) {
    
    const float r_max = -2.25f;
    const float r_min = 0.75f;
    const float c_max = 1.5f;
    const float c_min = -1.5f;
    
    const unsigned char palette[16][3]={
        {66, 30, 15},
        {25,7,26},
        {9,1,47},
        {4,4,73},
        {0,7,100},
        {12,44,138},
        {24,82,177},
        {57,125,209},
        {134,181,229},
        {221,236,248},
        {241,201,95},
        {255,170,0},
        {204,128,0},
        {153,87,0},
        {106,52,3}
    };

    int px = get_global_id(0);
    int py = get_global_id(1);
    if (px >= width || py >= height) return;

    float x0 = r_min + px * ((r_max - r_min) / width);
    float y0 = c_min + py * ((c_max - c_min) / height);
    unsigned int iteration;
    float x = 0.0f;
    float y = 0.0f;

    for (iteration = 0; iteration < iter_limit; iteration++) {
        float xn = x * x - y * y + x0;
        y = 2 * x * y + y0;
        x = xn;
        if (x * x + y * y > 2.0f) {
            break;
        }
    }
    int idx = width * py + px; 
    if (iteration == iter_limit){
        colors[idx][0] = 0;
        colors[idx][1] = 0;
        colors[idx][2] = 0;
        colors[idx][3] = 255;
    }else{
        x = iteration * 1.0 /iter_limit;
        int z = round(sinpi(x/2) * 16);
        colors[idx][0] = palette[z][0];
        colors[idx][1] = palette[z][1];
        colors[idx][2] = palette[z][2];
        colors[idx][3] = 255;
    }
    }
"#;

struct Interface_Opencl {
    kernel: Kernel,
    buffer_colors: Buffer<u8>,
    result: Vec<u8>,
}

impl Interface_Opencl {
    fn new(dims: (u32, u32)) -> Self {
        let dev = Device::first(Platform::first().unwrap()).unwrap();
        let context = ContextOCL::builder().build().unwrap();
        let que = Queue::new(&context, dev.clone(), None).unwrap();
        let prog = Program::builder()
            .src(KERNEL_SRC)
            .devices(dev)
            .build(&context)
            .unwrap();

        let buffer_colors = Buffer::builder()
            .queue(que.clone())
            .len((dims.0 * dims.1, 4))
            .fill_val(0u8)
            .flags(MemFlags::WRITE_ONLY)
            .build()
            .unwrap();

        let kernel = Kernel::builder()
            .program(&prog)
            .name("mandelbrot")
            .queue(que.clone())
            .global_work_size(dims)
            .arg(&buffer_colors)
            .arg(dims.0)
            .arg(dims.1)
            .arg(MAX_ITER)
            .build()
            .unwrap();

        Self {
            kernel,
            buffer_colors,
            result: vec![0u8; (dims.0 * dims.1 * 4)as usize],
        }
    }

    fn work(&mut self) {
        unsafe {
            self.kernel.enq().unwrap();
        }
        self.buffer_colors.read(&mut self.result).enq().unwrap();
    }

    fn read(&self) -> &Vec<u8> {
        &self.result
    }
}

struct App {
    worker: Interface_Opencl,
    dim: (u32, u32),
    grad: Gradient<LinSrgb>,
}

impl App {
    fn new(ctx: &mut Context, dim: (u32, u32)) -> Self {
        let mut worker = Interface_Opencl::new(dim);
        worker.work();
        Self {
            worker,
            dim,
            grad: build_grad(),
        }
    }
}

impl EventHandler for App {
    fn update(&mut self, ctx: &mut Context) -> GameResult {
        Ok(())
    }
    fn draw(&mut self, ctx: &mut Context) -> GameResult {
        graphics::clear(ctx, graphics::WHITE);
        let colors = self.worker.read();
        let height = self.dim.0;
        let width = self.dim.1;

        Image::from_rgba8(ctx, height as u16, width as u16, &colors).unwrap()
            .draw(ctx, DrawParam::new());
        graphics::present(ctx);

        Ok(())
    }
}

fn build_grad() -> Gradient<LinSrgb> {
    let mut palette = Vec::new();

    palette.push(LinSrgb::new(9.0, 1.0, 47.0));
    palette.push(LinSrgb::new(4.0, 4.0, 47.0));
    palette.push(LinSrgb::new(0.0, 7.0, 100.0));
    palette.push(LinSrgb::new(12.0, 44.0, 138.0));
    palette.push(LinSrgb::new(24.0, 82.0, 177.0));
    palette.push(LinSrgb::new(57.0, 125.0, 209.0));
    palette.push(LinSrgb::new(134.0, 181.0, 229.0));
    palette.push(LinSrgb::new(211.0, 236.0, 248.0));
    palette.push(LinSrgb::new(241.0, 233.0, 191.0));
    palette.push(LinSrgb::new(248.0, 201.0, 95.0));
    palette.push(LinSrgb::new(255.0, 170.0, 0.0));
    palette.push(LinSrgb::new(106.0, 52.0, 3.0));
    Gradient::new(palette)
}

fn color(iter: u32, buffer: &mut Rgba<u8>, grad: &Gradient<LinSrgb>) {
    if iter == MAX_ITER {
        buffer[0] = 0;
        buffer[1] = 0;
        buffer[2] = 0;
        buffer[3] = 255;
    } else {
        let x = (iter as f32 / MAX_ITER as f32 * 4.0 + 1.0).log2();
        let x = x / (4.0 + 1.0 as f32).log2();
        let color = grad.get(x).into_components();
        buffer[0] = color.0 as u8;
        buffer[1] = color.1 as u8;
        buffer[2] = color.2 as u8;
        buffer[3] = 255;
    }
}

fn main() {
    let (mut ctx, mut event_loop) = ContextBuilder::new("Mandelbrot Set", "Erkan")
        .window_mode(conf::WindowMode {
            width: 800.0,
            height: 800.0,
            maximized: false,
            resizable: false,
            ..Default::default()
        })
        .build()
        .unwrap();
    let mut app = App::new(&mut ctx, (800, 800));
    event::run(&mut ctx, &mut event_loop, &mut app);
}
