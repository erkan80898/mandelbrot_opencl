use num_complex::Complex;
use image::{RgbaImage,Rgba};
use palette::rgb::{LinSrgb};
use palette::gradient::Gradient;
use ggez::{Context, ContextBuilder,GameResult};
use ggez::conf;
use ggez::event::{self, EventHandler};
use ggez::graphics;
use ggez::graphics::{Canvas,Color,Drawable,Image};
use ggez::input::mouse::MouseButton;
use std::thread;
use std::sync::Arc;

const MAX_ITER:u32 = 256;

struct App{
    canvas:Canvas,
    image:Image,
}

impl App{

    fn new(ctx: &mut Context,dim: u16,buffer:RgbaImage)->Self{
        Self{canvas:
            Canvas::new(ctx,dim,dim,conf::NumSamples::Eight).unwrap(),
            image:Image::from_rgba8(ctx,dim,dim,&buffer.into_vec()).unwrap(),
        }
    }
}

impl EventHandler for App{

    fn update(&mut self, ctx: &mut Context) -> GameResult{
    
        Ok(())
    }
    fn draw(&mut self, ctx: &mut Context) -> GameResult{
        graphics::clear(ctx,graphics::WHITE);
        self.image.draw(ctx,graphics::DrawParam::new());
        graphics::present(ctx);
        Ok(())
    }
}

fn build_grad() -> Gradient<LinSrgb>{
    
    let mut palette = Vec::new();

    palette.push(LinSrgb::new(66.0,30.0,15.0));
    palette.push(LinSrgb::new(25.0,7.0,26.0));
    palette.push(LinSrgb::new(9.0,1.0,47.0));
    palette.push(LinSrgb::new(4.0,4.0,47.0));
    palette.push(LinSrgb::new(0.0,7.0,100.0));
    palette.push(LinSrgb::new(12.0,44.0,138.0));
    palette.push(LinSrgb::new(24.0,82.0,177.0));
    palette.push(LinSrgb::new(57.0,125.0,209.0));
    palette.push(LinSrgb::new(134.0,181.0,229.0));
    palette.push(LinSrgb::new(211.0,236.0,248.0));
    palette.push(LinSrgb::new(241.0,233.0,191.0));
    palette.push(LinSrgb::new(248.0,201.0,95.0));
    palette.push(LinSrgb::new(255.0,170.0,0.0));
    palette.push(LinSrgb::new(204.0,128.0,0.0));
    palette.push(LinSrgb::new(153.0,87.0,0.0));
    palette.push(LinSrgb::new(106.0,52.0,3.0));
    palette.push(LinSrgb::new(0.0,0.0,0.0));
    Gradient::new(palette)
}

fn color(complex:Complex<f32>,iter:u32,buffer:&mut Rgba<u8>,grad:&Gradient<LinSrgb>){
        let color = grad.get(iter as f32/MAX_ITER as f32).into_components();
        buffer[0] = color.0 as u8;
        buffer[1] = color.1 as u8;
        buffer[2] = color.2 as u8;
        buffer[3] = 255;
}

fn generate_mendel(dim:u32) -> RgbaImage{
    // Define img size, range of reals and i, and scale, iter range
    // It's found for 800x800, 256 iterations suffices
    let img_dim = dim;
    let r_min = -2.25f32;
    let r_max = 0.75f32;
    let c_min = -1.5f32;
    let c_max = 1.5f32;
    let iter_limit = MAX_ITER;

    let r_scale = (r_max - r_min) / img_dim as f32;
    let c_scale = (c_max - c_min) / img_dim as f32;

    let mut pixel_buffer = RgbaImage::new(img_dim,img_dim);
    let grad = build_grad();

    for (x,y,pixel) in pixel_buffer.enumerate_pixels_mut(){
        let r = x as f32 * r_scale + r_min;
        let c = y as f32 * c_scale + c_min;

        let c = Complex::new(r,c);

        let mut result = c;
        let mut iter = 0;
        while iter < iter_limit{
            result = result*result + c;
            if result.norm() > 2.0{
                break;
            }
            iter += 1;
        }
        color(c,iter,pixel,&grad);
    }
    return pixel_buffer;
}

fn main() {
    
    let mut data = RgbaImage::new(800,800);
    data = generate_mendel(800);

    let (mut ctx, mut event_loop) =
       ContextBuilder::new("Mandelbrot Set", "Erkan")
           .window_mode(conf::WindowMode{width:800.0,height:800.0,
               maximized:false,resizable:false,..Default::default()})
           .build()
           .unwrap();
    let mut app = App::new(&mut ctx,800,data);
    event::run(&mut ctx,&mut event_loop,&mut app);
}