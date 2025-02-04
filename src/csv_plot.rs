
use std::fs::File;
use std::io::{BufRead, BufReader};
use plotters::prelude::*;

pub fn plot_gradients(filename: &str, output_file: &str, save: bool) {
    if !save {
        return;
    }
    let file = File::open(filename).expect("Could not open file");
    let reader = BufReader::new(file);
    
    let mut data = Vec::new();
    for line in reader.lines() {
        let line = line.expect("Could not read line");
        let values: Vec<f64> = line
            .split(',')
            .map(|v| v.parse::<f64>().expect("Invalid number"))
            .collect();
        data.push((values[0], values[1]));
    }

    let root = BitMapBackend::new(output_file, (1024, 768)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let mut chart = ChartBuilder::on(&root)
        .caption("Gradient Norms Over Training", ("sans-serif", 40))
        .margin(10)
        .x_label_area_size(50)
        .y_label_area_size(50)
        .build_cartesian_2d(
            0f64..data.last().unwrap().0,
            0f64..data.iter().map(|(_, y)| *y).fold(0.0, f64::max),
        )
        .unwrap();

    chart.configure_mesh().draw().unwrap();

    chart
        .draw_series(LineSeries::new(data.into_iter(), &BLUE))
        .unwrap()
        .label("Gradient Norm")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    chart.configure_series_labels().border_style(&BLACK).draw().unwrap();
}


    // plot loss curve
    pub fn plot_loss_curve(loss_values: Vec<f64>, location: &str) -> Result<(), Box<dyn std::error::Error>> {
        let root = BitMapBackend::new(&location, (800, 600)).into_drawing_area();
        root.fill(&WHITE)?;

        let mut chart = ChartBuilder::on(&root)
            .caption("Loss Curve", ("sans-serif", 50))
            .build_cartesian_2d(0..loss_values.len(), 0.0..loss_values.iter().cloned().fold(f64::INFINITY, f64::min))?;

        chart.draw_series(LineSeries::new(
            loss_values.iter().enumerate().map(|(i, &loss)| (i, loss)),
            &BLUE,
        ))?;

        Ok(())
    }
