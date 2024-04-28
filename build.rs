fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_build::compile_protos("common/protobuf/grpc_service.proto")?;
    Ok(())
}
