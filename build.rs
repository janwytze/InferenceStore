fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_build::configure()
        .type_attribute(
            ".inference",
            "#[derive(serde::Serialize, serde::Deserialize)]",
        )
        .type_attribute(".inference", "#[serde(rename_all = \"camelCase\")]")
        .compile(
            &["common/protobuf/grpc_service.proto"],
            &["common/protobuf"],
        )?;

    Ok(())
}
