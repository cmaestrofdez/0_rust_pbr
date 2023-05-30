use cgmath::{Point3, Vector3};
use image::{codecs::png::PngEncoder, GenericImageView, ImageEncoder};
use itertools::{iproduct, Itertools};
use num_traits::float::FloatCore;
use rand::distributions::{Standard, Uniform};
use std::{cell::RefCell, fs::File, rc::Rc};

use crate::raytracerv2::Camera;
use crate::sampler::custom_rng;
use crate::{
    imagefilm::{ImgFilm },
    sampler::{Sampler, SamplerLd, SamplerLd2, SamplerUniform},
    samplerhalton::SamplerHalton,
};

use crate::Point2i;
#[test]
fn sample_test() {
    let mut sampler = SamplerUniform::new(((0, 0), (8, 8)),4, false, Some(0));
    sampler.start_pixel(Point2i::new(0,0));
    // println!("{:?}", sampler.get2d());
    println!("{:?}", sampler.get2d());
   while  sampler.start_next_sample(){
    println!("{:?}", sampler.get2d());
   }
   sampler.start_pixel(Point2i::new(1,1));
   println!("{:?}", sampler.get2d());
   while  sampler.start_next_sample(){
    println!("{:?}", sampler.get2d());
   }
    
}

#[test]
fn sample_test_rng(){
    let mut sampler = SamplerUniform::new(((0, 0), (8, 8)),4, false, Some(0));
    let mut supersumsampleraccum = 0.0;
    for y in 0..8 {
        for x in 0..8 {
            let p = Point2i::new(x as i64, y as i64);
            // println!("idx : {} pixel : {:?}", y * 16 + x ,  p);
            sampler.start_pixel(p);
            // println!("  {:?}", samplerhalton.get2d());
            let pair = sampler.get2d();
           
            while sampler.start_next_sample() {
                let mut sampleraccum = 0.0;
               
                for nextsample in 0..32 {
                    let newpair = sampler.get2d();
                    sampleraccum += newpair.0 + newpair.1;
                }
                supersumsampleraccum += sampleraccum;
                //  println!("   ->{:?}"  samplerhalton.gext2d().0);
            }
            let idx = x + 8 * y ;
            println!(" id {} ,{} ",idx, supersumsampleraccum);
          
        }
    }
    println!("  {} ", supersumsampleraccum);
}
fn mi_samplers_use1(sample: &mut Box<dyn Sampler>) {
    print!("{:?} ", sample.get_current_pixel());
    
    while sample.start_next_sample(){
        print!("{:?}  ", sample.get_current_pixel());
        sample.get2d();
    }
    //    sample.get_current_pixel();
    //    let s01d1 = &mut vec![0.0 ;1];
    //    sample.get_sample1d(s01d1,  &mut rand::thread_rng() );
    //    println!("{:?}", s01d1);
    //    println!("{:?}",  sample.has_samples() );
}

struct SampleWrapper {
    pub sample: Box<dyn Sampler>,
}
#[test]
fn sample_test2() {
    let mut sampler = SamplerUniform::new(((0, 0), (2, 2)),8, false, Some(0));
    let mut s = SampleWrapper {
        sample: Box::new(sampler),
    };

    mi_samplers_use1(&mut s.sample);
}

#[test]
fn sample_test3_ld() {
    let mut sld = SampleWrapper {
        sample: Box::new(SamplerLd::new(((0, 0), (2, 2)), 8)),
    }; 
    sld.sample.get_current_pixel();
}

#[test]
fn sample_test3_ld2() {
    {
        let sampler = &mut Box::new(SamplerLd2::new(((0, 0), (2, 2)), 8, false));
        sampler.start_pixel(Point2i::new(0, 0));
        let mut sampleidex = 0;
        while sampler.start_next_sample() {
            println!("{}", sampler.current_index_sample);
            sampleidex += 1;
        }
        assert_eq!(sampleidex, 8 - 1);
    }
    {
        let sampler = &mut Box::new(SamplerLd2::new(((0, 0), (2, 2)), 2046, false));
        for y in 0..2{
            for x in 0..2{
                sampler.start_pixel(Point2i::new(x, y));
                println!("{:?} , {}",sampler .get2d() , sampler.current_index_sample);
                
                while sampler.start_next_sample() {
                    println!("{:?}", sampler .get2d());
                   
                }
            }
        }
       
        
    }
}

fn render_write(filename: &str, pixels: &[u8], bounds: (u32, u32)) -> std::io::Result<()> {
    let fd = File::create(filename).unwrap();
    let encoder = PngEncoder::new(fd);
    encoder
        .write_image(pixels, bounds.0, bounds.1, image::ColorType::Rgb8)
        .unwrap();

    Ok(())
}

fn render_testfilm(
    film: Rc<RefCell<ImgFilm>>,
    sampler: &mut Box<dyn Sampler>,
) -> std::io::Result<()> {
    let f = film.to_owned();
    let w = f.borrow_mut().dims.1.x as usize;
    let h = f.borrow_mut().dims.1.y as usize;
    let camera = Camera::new(
        Point3::new(0.30, 1.31010, 2.5), // puede ser que a  medida que me acerco a y = 1 no funciona bien...esta bien la camara...no saldra allgun error con el vector up
        Point3::new(0.0, 0.0, 0.0),
        Vector3::new(0.0, 1.0, 0.0),
        90.0,
        1.0f64 / 1.0f64,
    );
    mod render_testfilm_test{
        fn lerp(x: f64, l: f64, h: f64) -> f64 {
            l + (h - l) * x
        }
        pub fn set_pixels(  w : usize, h : usize, pxls :&mut  Vec<u8>, samp : &(f64, f64)){
            let xpos = lerp(samp.0, 0.0, w as f64);
            let ypos = lerp(samp.1, 0.0, h as f64);
            let ixpos = xpos as usize;
            let iypos = ypos as usize;
            //    println!("  xpos, ypos,{:?},{:?}",    ixpos, iypos);
            pxls[(iypos * 3 * w) + ixpos * 3] += 1    ;
            pxls[(iypos * 3 * w) + ixpos * 3 + 1] = 1  ;
            pxls[(iypos * 3 * w) + ixpos * 3 + 2] = 1;
        }
        pub fn set_pixels_with_position(xpos :i64 , ypos : i64, w : usize, h : usize, pxls :&mut  Vec<u8>, samp : &(f64, f64)){
            let ixpos = xpos as usize;
            let iypos = ypos as usize;

            pxls[(iypos * 3 * w) + ixpos * 3] = 1    ;
            pxls[(iypos * 3 * w) + ixpos * 3 + 1] =1  ;
            pxls[(iypos * 3 * w) + ixpos * 3 + 2] = 1;
        }
    
    }
   






    let mut pixels = vec![0_u8; w * h * 3];
    let spp = sampler.get_spp();
    for (y, x) in iproduct!((0..h as i64), (0..w as i64)) {
       
        
        sampler.start_pixel(Point2i::new(x, y));
        let samp1 = sampler.get2d();
      //   println!("{:?}", samp1);
        render_testfilm_test::set_pixels_with_position( x, y, w, h,   & mut pixels, &samp1);
        while sampler.start_next_sample(){
            let samp2 = sampler.get2d();
           
            assert_eq!(samp2.0.floor() as i64, x);
            assert_eq!(samp2.1.floor() as i64, y);
            let sampnop = sampler.get2d();
            
            println!("{:?}", sampnop);
            render_testfilm_test::set_pixels_with_position( x, y, w, h,   & mut pixels, &samp2);
        }
        // for dim in 0..4096 {
        //     let samp1 = sampler.get2d();
        //     render_testfilm_test::set_pixels(  w, h,   & mut pixels, &samp1);
        // }
        

         
    }

    render_write(
        "mi_test/pixel_distribution_2.png",
        &pixels[..],
        (w as u32, h as u32),
    );
    Ok(())
}
  

#[test]
fn sample_test4_ld() {
    let h = 1024;
    let w = 1024;
    let film = Rc::new(RefCell::new(ImgFilm::from_filter_gauss(
        w,
        h,
        (2.0, 2.0),
        1.0,
    )));
    let mut sampler: Box<dyn Sampler> =
        Box::new(SamplerLd2::new(((0, 0), (w as u32, h as u32)), 2, false));
    render_testfilm(film, &mut sampler);
}

// https://github.com/wahn/rs_pbrt/blob/master/src/samplers/halton.rs
#[test]
fn sample_halton() {
    let rng = &mut custom_rng::CustomRng::new();
    let samplerhalton = &mut Box::new(SamplerHalton::new(
        &Point2i::new(0, 0),
        &Point2i::new(4, 4),
        4,
        false,
    ));
    let mut supersumsampleraccum = 0.0;
    for y in 0..4 {
        for x in 0..4 {
            let p = Point2i::new(x as i64, y as i64);
            // println!("idx : {} pixel : {:?}", y * 16 + x ,  p);
            samplerhalton.start_pixel(p);
            println!("{:?}", samplerhalton.get2d());

            let mut sampleraccum = 0.0;
            while samplerhalton.start_next_sample() {
                let pair = samplerhalton.get2d();
                println!("  --->{:?}", pair);
                // for nextsample in 0..100 {
                //     let newpair = samplerhalton.get2d();
                //     sampleraccum += newpair.0 + newpair.1;
                // }
                // sampleraccum += pair.0 + pair.1;
                //  println!("   ->{:?}"  samplerhalton.gext2d().0);
            }
            supersumsampleraccum += sampleraccum;
        }
    }
    println!("  {} ", supersumsampleraccum);
    return;
}
