use clap::Clap;
use ggez::conf;
use ggez::event::{self, EventHandler};
use ggez::graphics;
use ggez::graphics::{DrawParam, Drawable, Image};
use ggez::input::mouse;
use ggez::{Context, ContextBuilder, GameResult};
use ocl::{Buffer, Context as ContextOCL, Device, Kernel, MemFlags, Platform, Program, Queue};

const SCALE: f64 = 0.9;

static KERNEL_SRC: &'static str = r#"
    __kernel void mandelbrot(__global unsigned char colors[][4], double r_from,double r_to,
                        double c_from, double c_to, int width, int height,int iter_limit) {
        
    const unsigned char palette[16][3]={
        {25,7,26},
        {0,120,50},
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
        {106,52,3},
    };

    int px = get_global_id(0);
    int py = get_global_id(1);

    double x0 = r_from + px * (r_to - r_from) / width;
    double y0 = c_from + py * (c_to - c_from) / height;

    unsigned int iteration;
    double x = 0.0f;
    double y = 0.0f;

    for (iteration = 0; iteration < iter_limit; iteration++) {
        double xn = x * x - y * y + x0;
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
        int z = round(sinpi(x/2) * 15);
        colors[idx][0] = palette[z][0];
        colors[idx][1] = palette[z][1];
        colors[idx][2] = palette[z][2];
        colors[idx][3] = 255;
    }
    }
"#;

struct OpenCL {
    kernel: Kernel,
    buffer_colors: Buffer<u8>,
    result: Vec<u8>,
}

impl OpenCL{
    fn new(dims: (u32, u32),max_iter:u32) -> Self {
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
            .arg(-2.25f64)
            .arg(0.75f64)
            .arg(-1.5f64)
            .arg(1.5f64)
            .arg(dims.0)
            .arg(dims.1)
            .arg(max_iter)
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
    worker: OpenCL,
    dim: (u32, u32),
    complex: (f64,f64,f64,f64)
}

impl App {
    fn new(dim: (u32, u32),max_iter:u32) -> Self {
        let mut worker = OpenCL::new(dim,max_iter);
        worker.work();
        Self {
            worker,
            dim,
            complex:(-2.25,0.75,1.5,-1.5),
        }
    }
}

impl EventHandler for App {
    fn update(&mut self, _ctx: &mut Context) -> GameResult {
        Ok(())
    }

    fn draw(&mut self, ctx: &mut Context) -> GameResult{
        graphics::clear(ctx, graphics::WHITE);
        let colors = self.worker.read();

        Image::from_rgba8(ctx, self.dim.0 as u16, self.dim.1 as u16, &colors).unwrap()
            .draw(ctx, DrawParam::new())?;
    
        graphics::present(ctx)?;

        Ok(())
    }

    #[allow(unused_must_use)]
    fn mouse_wheel_event(&mut self, ctx: &mut Context, _x: f32, _y: f32){
        let kernel = &self.worker.kernel;

        let unit_r = (self.complex.1 - self.complex.0) / self.dim.0 as f64;
        let unit_c = (self.complex.3 - self.complex.2) / self.dim.1 as f64;

        let plane_point = (self.complex.0 + unit_r* mouse::position(ctx).x as f64,
                                        self.complex.2 + unit_c * mouse::position(ctx).y as f64);

        self.complex.0 = plane_point.0 - self.dim.0 as f64 / 2.0 * unit_r * SCALE;
        self.complex.1 = plane_point.0 + self.dim.0 as f64 / 2.0 * unit_r * SCALE;
        self.complex.2 = plane_point.1 - self.dim.1 as f64 / 2.0 * unit_c * SCALE;
        self.complex.3 = plane_point.1 + self.dim.1 as f64 / 2.0 * unit_c * SCALE;

        kernel.set_arg(1,self.complex.0);
        kernel.set_arg(2,self.complex.1);
        kernel.set_arg(3,self.complex.2);
        kernel.set_arg(4,self.complex.3);        
    
        self.worker.work();
    }
}

#[derive(Clap)]
#[clap(version = "1.0", author = "Erkan U. <erkan808987@gmail.com>")]
struct Opts{
    height: u32,
    width: u32,
    iteration: u32,
}

#[allow(unused_must_use)]
fn main() {
    let opts: Opts = Opts::parse();

    let (mut ctx, mut event_loop) = ContextBuilder::new("Mandelbrot Set", "Erkan")
        .window_mode(conf::WindowMode {
            width: opts.width as f32,
            height: opts.height as f32,
            maximized: false,
            resizable: false,
            ..Default::default()
        })
        .build()
        .unwrap();
    let mut app = App::new((opts.width, opts.height),opts.iteration);
    event::run(&mut ctx, &mut event_loop, &mut app);
}
